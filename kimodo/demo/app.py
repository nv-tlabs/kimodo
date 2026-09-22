# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import shutil
import threading
import time
from typing import Optional

import numpy as np
import torch

import viser
from kimodo.assets import DEMO_ASSETS_ROOT
from kimodo.device_utils import resolve_torch_device
from kimodo.model.load_model import load_model
from kimodo.model.registry import resolve_model_name
from kimodo.skeleton import SkeletonBase, SOMASkeleton30
from kimodo.tools import load_json
from kimodo.viz import viser_utils
from kimodo.viz.viser_utils import (
    Character,
    CharacterMotion,
    EEJointsKeyframeSet,
    FullbodyKeyframeSet,
    RootKeyframe2DSet,
)
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from . import generation, ui
from .config import (
    DARK_THEME,
    DEFAULT_CUR_DURATION,
    DEFAULT_MODEL,
    DEFAULT_PLAYBACK_SPEED,
    DEFAULT_PROMPT,
    DEMO_UI_QUICK_START_MODAL_MD,
    EXAMPLES_ROOT_DIR,
    HF_MODE,
    LIGHT_THEME,
    MAX_ACTIVE_USERS,
    MAX_DURATION,
    MAX_SESSION_MINUTES,
    MIN_DURATION,
    MODEL_EXAMPLES_DIRS,
    MODEL_NAMES,
    SERVER_NAME,
    SERVER_PORT,
)
from .embedding_cache import CachedTextEncoder
from .queue_manager import QueueManager, UserQueue
from .state import ClientSession, ModelBundle


class Demo:
    def __init__(self, default_model_name: str = DEFAULT_MODEL):
        self.device = resolve_torch_device("auto")
        print(f"Using device: {self.device}")
        self.models: dict[str, ModelBundle] = {}
        self._text_encoder = None
        resolved = resolve_model_name(default_model_name, "Kimodo")
        if resolved not in MODEL_NAMES:
            raise ValueError(f"Unknown model '{default_model_name}'. Expected one of: {MODEL_NAMES}")
        self.default_model_name = resolved
        self.ensure_examples_layout()
        self.load_model(self.default_model_name)

        # Serialize GPU-bound generation across all clients
        self._generation_lock = threading.Lock()
        self._cuda_healthy = True

        # Per-client sessions
        self.client_sessions: dict[int, ClientSession] = {}
        self.start_direction_markers: dict[int, viser_utils.WaypointMesh] = {}
        self.grid_handles: dict[int, viser.GridHandle] = {}

        self.server = viser.ViserServer(
            host=SERVER_NAME,
            port=SERVER_PORT,
            label="Kimodo",
            enable_camera_keyboard_controls=False,  # don't move the camera with the arrow keys
        )
        self.server.scene.world_axes.visible = False  # used for debugging
        self.server.scene.set_up_direction("+y")

        # Register callbacks for session handling
        self.server.on_client_connect(self.on_client_connect)
        self.server.on_client_disconnect(self.on_client_disconnect)

        # HF mode: queue and session limit
        if HF_MODE:
            self.user_queue = UserQueue(MAX_ACTIVE_USERS, MAX_SESSION_MINUTES)
            self.queue_manager = QueueManager(
                queue=self.user_queue,
                server=self.server,
                setup_demo_for_client=self._setup_demo_for_client,
                cleanup_session=self._cleanup_session_for_client,
            )
        else:
            self.user_queue = None
            self.queue_manager = None

        # create grid and floor
        self.floor_len = 20.0  # meters

    def ensure_examples_layout(self) -> None:
        os.makedirs(EXAMPLES_ROOT_DIR, exist_ok=True)
        for model_dir in MODEL_EXAMPLES_DIRS.values():
            os.makedirs(model_dir, exist_ok=True)

        for entry in os.listdir(EXAMPLES_ROOT_DIR):
            if entry in MODEL_EXAMPLES_DIRS:
                continue
            src = os.path.join(EXAMPLES_ROOT_DIR, entry)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(
                MODEL_EXAMPLES_DIRS.get(DEFAULT_MODEL, next(iter(MODEL_EXAMPLES_DIRS.values()))),
                entry,
            )
            if not os.path.exists(dst):
                shutil.move(src, dst)

    def get_examples_base_dir(self, model_name: str, absolute: bool = True) -> str:
        return MODEL_EXAMPLES_DIRS[model_name]

    def load_model(self, model_name: str) -> ModelBundle:
        if model_name in self.models:
            return self.models[model_name]

        print(f"Loading model {model_name}...")
        try:
            model = load_model(
                modelname=model_name,
                device=self.device,
                text_encoder=self._text_encoder,
            )
        except Exception as e:
            print(f"Error loading model: {e}\nMake sure text encoder server is running!")
            raise e

        if hasattr(model, "text_encoder"):
            if self._text_encoder is None:
                self._text_encoder = model.text_encoder
            model.text_encoder = CachedTextEncoder(model.text_encoder, model_name=model_name)

        skeleton = model.motion_rep.skeleton
        if isinstance(skeleton, SOMASkeleton30):
            skeleton = skeleton.somaskel77.to(model.device)
        bundle = ModelBundle(
            model=model,
            motion_rep=model.motion_rep,
            skeleton=skeleton,
            model_fps=model.motion_rep.fps,
        )
        self.models[model_name] = bundle
        print(f"Model {model_name} loaded successfully")
        self.prewarm_embedding_cache(model_name, bundle.model)
        return bundle

    def prewarm_embedding_cache(self, model_name: str, model: object) -> None:
        encoder = getattr(model, "text_encoder", None)
        if not isinstance(encoder, CachedTextEncoder):
            return

        prompt_set = set()
        prompt_set.add(DEFAULT_PROMPT)

        examples_dir = MODEL_EXAMPLES_DIRS.get(model_name)
        if examples_dir and os.path.isdir(examples_dir):
            for entry in os.listdir(examples_dir):
                example_dir = os.path.join(examples_dir, entry)
                if not os.path.isdir(example_dir):
                    continue
                meta_path = os.path.join(example_dir, "meta.json")
                if not os.path.exists(meta_path):
                    continue
                try:
                    meta = load_json(meta_path)
                except Exception:
                    continue
                for prompt in meta.get("prompts_text", []):
                    if isinstance(prompt, str):
                        prompt_set.add(prompt)

        if prompt_set:
            encoder.prewarm(list(prompt_set))

    def build_constraint_tracks(
        self, client: viser.ClientHandle, skeleton: SkeletonBase
    ) -> dict[str, viser_utils.ConstraintSet]:
        return {
            "Full-Body": FullbodyKeyframeSet(
                name="Full-Body",
                server=client,
                skeleton=skeleton,
            ),
            "End-Effectors": EEJointsKeyframeSet(
                name="End-Effectors",
                server=client,
                skeleton=skeleton,
            ),
            "2D Root": RootKeyframe2DSet(
                name="2D Root",
                server=client,
                skeleton=skeleton,
            ),
        }

    def set_timeline_defaults(self, timeline, model_fps: float) -> None:
        timeline.set_defaults(
            default_text=DEFAULT_PROMPT,
            default_duration=int(DEFAULT_CUR_DURATION * model_fps - 1),
            min_duration=int(MIN_DURATION * model_fps - 1),  # 2 seconds minimum,
            max_duration=int(
                MAX_DURATION * model_fps - 1  # - NB_TRANSITION_FRAMES
            ),  # 10 seconds maximum, minus the transition frames, if needed
            default_num_frames_zoom=int(1.10 * 10 * model_fps),  # a bit more than the max
            max_frames_zoom=1000,
            fps=model_fps,
        )

    def _apply_constraint_overlay_visibility(self, session: ClientSession) -> None:
        """Apply show-all vs show-only-current-frame to constraint overlays."""
        only_frame = session.frame_idx if session.show_only_current_constraint else None
        for constraint in session.constraints.values():
            constraint.set_overlay_visibility(only_frame)

    def set_constraint_tracks_visible(self, session: ClientSession, visible: bool) -> None:
        timeline = session.client.timeline
        timeline_data = session.timeline_data
        if timeline_data.get("constraint_tracks_visible", True) == visible:
            return

        with timeline_data["keyframe_update_lock"]:
            if visible:
                for track_id, track_info in timeline_data["tracks"].items():
                    timeline.add_track(
                        track_info["name"],
                        track_type=track_info.get("track_type", "keyframe"),
                        color=track_info.get("color"),
                        height_scale=track_info.get("height_scale", 1.0),
                        uuid=track_id,
                    )

                for keyframe_id, keyframe_data in timeline_data["keyframes"].items():
                    timeline.add_keyframe(
                        track_id=keyframe_data["track_id"],
                        frame=keyframe_data["frame"],
                        value=keyframe_data.get("value"),
                        opacity=keyframe_data.get("opacity", 1.0),
                        locked=keyframe_data.get("locked", False),
                        uuid=keyframe_id,
                    )

                for interval_id, interval_data in timeline_data["intervals"].items():
                    timeline.add_interval(
                        track_id=interval_data["track_id"],
                        start_frame=interval_data["start_frame_idx"],
                        end_frame=interval_data["end_frame_idx"],
                        value=interval_data.get("value"),
                        opacity=interval_data.get("opacity", 1.0),
                        locked=interval_data.get("locked", False),
                        uuid=interval_id,
                    )
            else:
                for track_id in list(timeline_data["tracks"].keys()):
                    timeline.remove_track(track_id)

        timeline_data["constraint_tracks_visible"] = visible

    def _cleanup_session_for_client(self, client_id: int) -> None:
        """Remove session and scene state for a client (e.g. on session expiry)."""
        if client_id in self.client_sessions:
            del self.client_sessions[client_id]
        self.start_direction_markers.pop(client_id, None)
        self.grid_handles.pop(client_id, None)

    def _setup_demo_for_client(self, client: viser.ClientHandle) -> None:
        """Initialize scene, GUI, and session state for a client (no modals)."""
        self.setup_scene(client)

        model_bundle = self.load_model(self.default_model_name)

        # Initialize each empty constraint track
        constraint_tracks = self.build_constraint_tracks(client, model_bundle.skeleton)

        # Create GUI elements for this client
        (
            gui_elements,
            timeline_tracks,
            example_dict,
            gui_examples_dropdown,
            gui_save_example_path_text,
            gui_model_selector,
        ) = ui.create_gui(
            demo=self,
            client=client,
            model_name=self.default_model_name,
            model_fps=model_bundle.model_fps,
        )
        timeline_data = {
            "tracks": timeline_tracks,
            "tracks_ids": {val["name"]: key for key, val in timeline_tracks.items()},
            "keyframes": {},
            "intervals": {},
            "keyframe_update_lock": threading.Lock(),
            "keyframe_move_timers": {},
            "pending_keyframe_moves": {},  # keyframe_id -> new_frame
            "constraint_tracks_visible": True,
            "dense_path_after_release_timer": None,
        }

        # Initialize session state
        cur_duration = DEFAULT_CUR_DURATION
        max_frame_idx = int(cur_duration * model_bundle.model_fps - 1)

        session = ClientSession(
            client=client,
            gui_elements=gui_elements,
            motions={},
            constraints=constraint_tracks,
            timeline_data=timeline_data,
            frame_idx=0,
            playing=False,
            playback_speed=DEFAULT_PLAYBACK_SPEED,
            cur_duration=cur_duration,
            max_frame_idx=max_frame_idx,
            updating_motions=False,
            edit_mode=False,
            model_name=self.default_model_name,
            model_fps=model_bundle.model_fps,
            skeleton=model_bundle.skeleton,
            motion_rep=model_bundle.motion_rep,
            examples_base_dir=self.get_examples_base_dir(self.default_model_name, absolute=True),
            example_dict=example_dict,
            gui_examples_dropdown=gui_examples_dropdown,
            gui_save_example_path_text=gui_save_example_path_text,
            gui_model_selector=gui_model_selector,
        )

        self.client_sessions[client.client_id] = session

        # Initialize default character for this client
        self.add_character_motion(client, session.skeleton)

    def on_client_connect(self, client: viser.ClientHandle) -> None:
        """Initialize GUI and state for each new client."""
        print(f"Client {client.client_id} connected")

        if HF_MODE and self.queue_manager is not None:
            self.queue_manager.on_client_connect(client)
        else:
            # Show quick start popup when a browser client connects (non-HF mode).
            with client.gui.add_modal(
                "Welcome — Quick Start",
                size="xl",
                show_close_button=True,
                save_choice="kimodo.demo.quick_start_ack",
            ) as modal:
                client.gui.add_markdown(DEMO_UI_QUICK_START_MODAL_MD)
                client.gui.add_button("Got it (don't remind me again)").on_click(lambda _event: modal.close())
            self._setup_demo_for_client(client)

    def setup_scene(self, client: viser.ClientHandle) -> None:
        self.configure_theme(client)
        client.camera.position = np.array(
            [2.7417358737841426, 1.8790455698853281, 7.675741569777456],
            dtype=np.float64,
        )
        client.camera.look_at = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        client.camera.up_direction = np.array(
            [-1.1102230246251568e-16, 1.0, 1.3596310734468913e-32],
            dtype=np.float64,
        )
        client.camera.fov = np.deg2rad(45.0)
        grid_handle = client.scene.add_grid(
            "/grid",
            width=self.floor_len,
            height=self.floor_len,
            wxyz=viser.transforms.SO3.from_x_radians(-np.pi / 2.0).wxyz,
            position=(0.0, 0.0001, 0.0),
            fade_distance=3 * self.floor_len,
            section_color=LIGHT_THEME["grid"],
            infinite_grid=True,
        )
        self.grid_handles[client.client_id] = grid_handle
        # marker for origin
        origin_waypoint = viser_utils.WaypointMesh(
            "/origin_waypoint",
            client,
            position=np.array([0.0, 0.0, 0.0]),
            heading=np.array([0.0, 1.0]),
            color=(0, 0, 255),
        )
        self.start_direction_markers[client.client_id] = origin_waypoint

    def on_client_disconnect(self, client: viser.ClientHandle) -> None:
        """Clean up when client disconnects."""
        print(f"Client {client.client_id} disconnected")
        client_id = client.client_id

        if HF_MODE and self.queue_manager is not None:
            self.queue_manager.on_client_disconnect(client_id)

        self._cleanup_session_for_client(client_id)

    def set_start_direction_visible(self, client_id: int, visible: bool) -> None:
        marker = self.start_direction_markers.get(client_id)
        if marker is None:
            return
        marker.set_visible(visible)

    def client_active(self, client_id: int) -> bool:
        return client_id in self.client_sessions

    def add_character_motion(
        self,
        client: viser.ClientHandle,
        skeleton: SkeletonBase,
        joints_pos: Optional[torch.Tensor] = None,
        joints_rot: Optional[torch.Tensor] = None,
        foot_contacts: Optional[torch.Tensor] = None,
    ) -> None:
        client_id = client.client_id
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]

        ci = len(session.motions)
        character_name = f"character{ci}"
        # build character skeleton and skinning mesh
        if "g1" in session.model_name:
            mesh_mode = "g1_stl"
        elif "smplx" in session.model_name:
            mesh_mode = "smplx_skin"
        elif "soma" in session.model_name:
            if session.gui_elements.gui_use_soma_layer_checkbox.value:
                mesh_mode = "soma_layer_skin"
            else:
                mesh_mode = "soma_skin"
        else:
            raise ValueError("The model name is not recognized for skinning.")

        new_character = Character(
            character_name,
            client,
            skeleton,
            create_skeleton_mesh=True,
            create_skinned_mesh=True,
            visible_skeleton=False,  # don't show immediately
            visible_skinned_mesh=False,  # don't show immediately
            skinned_mesh_opacity=session.gui_elements.gui_viz_skinned_mesh_opacity_slider.value,
            show_foot_contacts=session.gui_elements.gui_viz_foot_contacts_checkbox.value,
            dark_mode=session.gui_elements.gui_dark_mode_checkbox.value,
            mesh_mode=mesh_mode,
            gui_use_soma_layer_checkbox=session.gui_elements.gui_use_soma_layer_checkbox,
        )

        # if no motion given, initialize to character default (rest) pose for one frame
        init_joints_pos, init_joints_rot = new_character.get_pose()
        if joints_pos is None:
            joints_pos = init_joints_pos[None].repeat(session.max_frame_idx + 1, 1, 1)
        if joints_rot is None:
            joints_rot = init_joints_rot[None].repeat(session.max_frame_idx + 1, 1, 1, 1)

        new_motion = CharacterMotion(new_character, joints_pos, joints_rot, foot_contacts)
        # save the motion in our dict
        session.motions[character_name] = new_motion

        # put the character at the right frame
        new_motion.set_frame(session.frame_idx)

        # put them visible with a small delay
        # so that the set_frame function has time to finish
        def _set_visibility():
            new_motion.character.set_skinned_mesh_visibility(session.gui_elements.gui_viz_skinned_mesh_checkbox.value)
            new_motion.character.set_skeleton_visibility(session.gui_elements.gui_viz_skeleton_checkbox.value)

        timer = threading.Timer(
            0.2,  # 0.2s delay
            _set_visibility,
        )
        timer.start()

    def clear_motions(self, client_id: int) -> None:
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        for motion in list(session.motions.values()):
            motion.clear()
        session.motions.clear()

    def compute_model_constraints_lst(
        self,
        session: ClientSession,
        model_bundle: ModelBundle,
        num_frames: int,
    ):
        return generation.compute_model_constraints_lst(session, model_bundle, num_frames, self.device)

    def check_cuda_health(self) -> bool:
        """Check if CUDA is still functional.

        Trigger auto-restart if corrupted.
        """
        if self.device == "cpu":
            return True
        try:
            torch.tensor([1.0], device=self.device) + torch.tensor([1.0], device=self.device)
            return True
        except RuntimeError as e:
            if "device-side assert" in str(e) or "CUDA error" in str(e):
                if self._cuda_healthy:
                    self._cuda_healthy = False
                    print("FATAL: CUDA context is corrupted (device-side assert). " "The process must be restarted.")
                    self._trigger_restart()
                return False
            raise

    def _trigger_restart(self) -> None:
        """Exit the process so the HF Space (or systemd/Docker) can restart it."""
        import sys

        print("Initiating automatic restart due to unrecoverable CUDA error...")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    def generate(
        self,
        client: viser.ClientHandle,
        prompts: list[str],
        num_frames: list[int],
        num_samples: int,
        seed: int,
        diffusion_steps: int,
        cfg_weight: Optional[list[float]] = None,
        cfg_type: Optional[str] = None,
        postprocess_parameters: Optional[dict] = None,
        transitions_parameters: Optional[dict] = None,
        real_robot_rotations: bool = False,
    ) -> None:
        if not self._cuda_healthy:
            raise RuntimeError("CUDA is in a corrupted state. The space is restarting...")

        locked = self._generation_lock.acquire(blocking=False)
        if not locked:
            waiting_notif = client.add_notification(
                title="Waiting for GPU...",
                body="Another generation is in progress. Yours will start automatically.",
                loading=True,
                with_close_button=False,
            )
            self._generation_lock.acquire()
            waiting_notif.remove()

        try:
            session = self.client_sessions[client.client_id]
            model_bundle = self.load_model(session.model_name)
            generation.generate(
                client=client,
                session=session,
                model_bundle=model_bundle,
                prompts=prompts,
                num_frames=num_frames,
                num_samples=num_samples,
                seed=seed,
                diffusion_steps=diffusion_steps,
                cfg_weight=cfg_weight,
                cfg_type=cfg_type,
                postprocess_parameters=postprocess_parameters,
                transitions_parameters=transitions_parameters,
                real_robot_rotations=real_robot_rotations,
                device=self.device,
                clear_motions=self.clear_motions,
                add_character_motion=self.add_character_motion,
            )
        finally:
            self._generation_lock.release()

    def set_frame(self, client_id: int, frame_idx: int, update_timeline: bool = True):
        if not self.client_active(client_id):
            return

        session = self.client_sessions[client_id]

        session.frame_idx = frame_idx
        if update_timeline:
            session.client.timeline.set_current_frame(frame_idx)
        for motion in list(session.motions.values()):
            motion.set_frame(frame_idx)
        self._apply_constraint_overlay_visibility(session)

    def run(self) -> None:
        update_counter = 0
        cuda_check_interval = 300
        while True:
            last_update_time = time.time()
            if self.models:
                # the max playback speed is 2x the model fps (from gui_playback_speed_buttons)
                playback_fps = max(bundle.model_fps for bundle in self.models.values()) * 2.0
            else:
                playback_fps = 60.0

            # update each client session independently
            #   copy to a list first to avoid changing size if client disconnects
            for client_id, session in list(self.client_sessions.items()):
                update_interval = int(playback_fps / (session.playback_speed * session.model_fps))
                new_frame_idx = session.frame_idx
                if session.playing and update_counter % update_interval == 0:
                    if session.frame_idx >= session.max_frame_idx:
                        new_frame_idx = 0
                    else:
                        new_frame_idx = session.frame_idx + 1

                    # make sure the client is still active before updating the frame
                    if self.client_active(client_id):
                        self.set_frame(client_id, new_frame_idx)

            if update_counter % cuda_check_interval == 0:
                self.check_cuda_health()

            time_remaining = max(0, 1.0 / playback_fps - (time.time() - last_update_time))
            time.sleep(time_remaining)
            update_counter += 1
            update_counter %= playback_fps  # wrap around to 0 every second

    def configure_theme(
        self,
        client: viser.ClientHandle,
        dark_mode: bool = False,
        titlebar_dark_mode_checkbox_uuid: str | None = None,
    ):
        # Sync grid color with theme (light vs dark)
        theme = DARK_THEME if dark_mode else LIGHT_THEME
        grid_handle = self.grid_handles.get(client.client_id)
        if grid_handle is not None:
            grid_handle.section_color = theme["grid"]

        #
        # setup theme
        #
        buttons = (
            TitlebarButton(
                text="Documentation",
                icon="Description",
                href="https://research.nvidia.com/labs/sil/projects/kimodo/docs/interactive_demo/index.html",
            ),
            TitlebarButton(
                text="Project Page",
                icon=None,
                href="https://research.nvidia.com/labs/sil/projects/kimodo/",
            ),
            TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/nv-tlabs/kimodo",
            ),
        )
        assets_dir = DEMO_ASSETS_ROOT
        logo_light_path = assets_dir / "nvidia_logo.png"
        logo_dark_path = assets_dir / "nvidia_logo_dark.png"
        if logo_light_path.exists():
            light_b64 = base64.standard_b64encode(logo_light_path.read_bytes()).decode("ascii")
            dark_b64 = (
                base64.standard_b64encode(logo_dark_path.read_bytes()).decode("ascii")
                if logo_dark_path.exists()
                else None
            )
            image = TitlebarImage(
                image_url_light=f"data:image/png;base64,{light_b64}",
                image_url_dark=(f"data:image/png;base64,{dark_b64}" if dark_b64 else None),
                image_alt="NVIDIA",
                href="https://www.nvidia.com/",
            )
        else:
            image = None
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image, title_text="Kimodo")
        client.gui.set_panel_label("Kimodo")
        client.gui.configure_theme(
            titlebar_content=titlebar_theme,
            control_layout="floating",  # "floating",  # ['floating', 'collapsible', 'fixed']
            control_width="large",  # ['small', 'medium', 'large']
            dark_mode=dark_mode,
            show_logo=False,  # hide viser logo on bottom left corner
            show_share_button=False,
            titlebar_dark_mode_checkbox_uuid=titlebar_dark_mode_checkbox_uuid,
            brand_color=(152, 189, 255),  # (60, 131, 0),  # (R, G, B) tuple
        )
