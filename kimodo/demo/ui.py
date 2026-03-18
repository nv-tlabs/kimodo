# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001
import math
import os
import threading
from typing import Optional

from kimodo.constraints import load_constraints_lst, save_constraints_lst
from kimodo.exports.bvh import motion_to_bvh_bytes
from kimodo.tools import to_torch
from kimodo.viz import viser_utils
from kimodo.viz.viser_utils import GuiElements
import numpy as np
import torch
import viser
from viser._timeline_api import PROMPT_COLORS

from . import generation
from .config import (
    DEFAULT_CUR_DURATION,
    DEMO_UI_INSTRUCTIONS_TAB_MD,
    get_datasets,
    get_model_info,
    get_models_for_dataset_skeleton,
    get_skeleton_display_name,
    get_skeleton_display_names_for_dataset,
    get_skeleton_key_from_display_name,
    get_short_key_from_display_name,
    HF_MODE,
    INIT_POSTPROCESSING,
    MODEL_NAMES,
    NB_TRANSITION_FRAMES,
    SHOW_TRANSITION_PARAMS,
)
from .state import ClientSession
from kimodo.skeleton import G1Skeleton34


def extract_intervals_and_singles(t: torch.Tensor):
    intervals = []
    intervals_indices = []
    single_frames = []
    single_frames_indices = []

    start_idx = 0

    for i in range(1, len(t) + 1):
        # End of run if:
        #  - end of tensor
        #  - non-consecutive value
        if i == len(t) or t[i] != t[i - 1] + 1:
            run_length = i - start_idx

            if run_length >= 2:
                intervals.append((int(t[start_idx]), int(t[i - 1])))
                intervals_indices.append((start_idx, i - 1))
            else:
                single_frames.append(int(t[start_idx]))
                single_frames_indices.append(start_idx)

            start_idx = i

    return intervals, intervals_indices, single_frames, single_frames_indices


def create_gui(
    demo,
    client: viser.ClientHandle,
    model_name: str,
    model_fps: float,
):
    """Create GUI elements for a specific client."""
    client_id = client.client_id

    def get_active_session(event_client: viser.ClientHandle | None):
        if event_client is None:
            return None
        if not demo.client_active(event_client.client_id):
            return None
        return demo.client_sessions[event_client.client_id]

    def _get_download_format_options(*, is_g1: bool) -> list[str]:
        options = ["NPZ", "BVH"]
        if is_g1:
            options.append("CSV")
        return options

    def _update_download_format_options(dropdown: viser.GuiInputHandle, *, is_g1: bool) -> None:
        options = _get_download_format_options(is_g1=is_g1)
        current_value = str(dropdown.value).upper() if dropdown.value is not None else "NPZ"
        dropdown.options = options
        dropdown.value = current_value if current_value in options else "NPZ"

    def build_timeline_tracks():
        timeline = client.timeline
        demo.set_timeline_defaults(timeline, model_fps)
        timeline.set_visible(True)
        timeline.set_current_frame(0)

        timeline_tracks = {}
        fullbody_id = timeline.add_track(
            "Full-Body",
            track_type="keyframe",
            color=(219, 148, 86),
            height_scale=0.5,
        )
        timeline_tracks[fullbody_id] = {
            "name": "Full-Body",
            "track_type": "keyframe",
            "color": (219, 148, 86),
            "height_scale": 0.5,
        }

        root2d_id = timeline.add_track(
            "2D Root",
            track_type="keyframe",
            color=(150, 100, 200),
            height_scale=0.5,
        )
        timeline_tracks[root2d_id] = {
            "name": "2D Root",
            "track_type": "keyframe",
            "color": (150, 100, 200),
            "height_scale": 0.5,
        }
        lefthand_id = timeline.add_track(
            "Left Hand",
            track_type="keyframe",
            color=(100, 200, 150),
            height_scale=0.5,
        )
        timeline_tracks[lefthand_id] = {
            "name": "Left Hand",
            "track_type": "keyframe",
            "color": (100, 200, 150),
            "height_scale": 0.5,
        }
        righthand_id = timeline.add_track(
            "Right Hand",
            track_type="keyframe",
            color=(200, 100, 150),
            height_scale=0.5,
        )
        timeline_tracks[righthand_id] = {
            "name": "Right Hand",
            "track_type": "keyframe",
            "color": (200, 100, 150),
            "height_scale": 0.5,
        }
        leftfoot_id = timeline.add_track(
            "Left Foot",
            track_type="keyframe",
            color=(219, 148, 86),
            height_scale=0.5,
        )
        timeline_tracks[leftfoot_id] = {
            "name": "Left Foot",
            "track_type": "keyframe",
            "color": (219, 148, 86),
            "height_scale": 0.5,
        }
        rightfoot_id = timeline.add_track(
            "Right Foot",
            track_type="keyframe",
            color=(150, 100, 200),
            height_scale=0.5,
        )
        timeline_tracks[rightfoot_id] = {
            "name": "Right Foot",
            "track_type": "keyframe",
            "color": (150, 100, 200),
            "height_scale": 0.5,
        }
        return timeline, timeline_tracks

    timeline, timeline_tracks = build_timeline_tracks()
    # These handles are part of GuiElements, but the demo currently uses timeline + buttons
    # embedded in the Viser UI instead of custom controls.
    gui_play_pause_button = None
    gui_next_frame_button = None
    gui_prev_frame_button = None
    gui_timeline = None
    gui_duration_slider = None

    # now other gui elements
    tab_group = client.gui.add_tab_group()

    #
    # Playback and Motion generation controls
    #
    with tab_group.add_tab("Generate", viser.Icon.WALK):
        with client.gui.add_folder("Model Selection", expand_by_default=True):
            info = get_model_info(model_name)
            if info is None:
                info = get_model_info(next(iter(MODEL_NAMES)))

            def get_allowed_skeleton_labels(dataset_ui_label: str) -> list[str]:
                labels = get_skeleton_display_names_for_dataset(dataset_ui_label, family="Kimodo")
                if HF_MODE:
                    labels = [label for label in labels if get_skeleton_key_from_display_name(label) != "SMPLX"]
                return labels

            dataset_ui_label = "Rigplay" if HF_MODE else info.dataset_ui_label
            datasets = ["Rigplay"] if HF_MODE else get_datasets(family="Kimodo")
            skeleton_labels = get_allowed_skeleton_labels(dataset_ui_label)
            initial_skeleton_label = get_skeleton_display_name(info.skeleton)
            if initial_skeleton_label not in skeleton_labels and skeleton_labels:
                initial_skeleton_label = skeleton_labels[0]
            initial_skeleton_key = (
                get_skeleton_key_from_display_name(initial_skeleton_label) if skeleton_labels else None
            )
            models_for_pair = (
                get_models_for_dataset_skeleton(dataset_ui_label, initial_skeleton_key, family="Kimodo")
                if initial_skeleton_key is not None
                else []
            )
            version_options = [m.display_name for m in models_for_pair]
            initial_version = (
                info.display_name
                if info.display_name in version_options
                else (version_options[0] if version_options else "")
            )
            gui_dataset_selector = client.gui.add_dropdown(
                "Training dataset",
                options=datasets,
                initial_value=dataset_ui_label,
                visible=not HF_MODE,
            )
            gui_skeleton_selector = client.gui.add_dropdown(
                "Model" if HF_MODE else "Skeleton",
                options=skeleton_labels,
                initial_value=initial_skeleton_label,
            )
            gui_version_selector = client.gui.add_dropdown(
                "Version",
                options=version_options,
                initial_value=initial_version,
            )
            gui_version_selector.visible = len(models_for_pair) > 1
            gui_model_display = client.gui.add_markdown(
                content=f"**Model:** {initial_version}",
            )
            gui_load_model_button = client.gui.add_button(
                "Load model",
                hint="Load the selected model (dataset, skeleton, version).",
            )

            class ModelSelectorHandle:
                """Wrapper so session and callbacks can treat three dropdowns as one."""

                def __init__(self):
                    self._dataset = gui_dataset_selector
                    self._skeleton = gui_skeleton_selector
                    self._version = gui_version_selector
                    self._display = gui_model_display

                @property
                def value(self) -> str:
                    return get_short_key_from_display_name(self._version.value) or ""

                def set_from_short_key(self, short_key: str) -> None:
                    info = get_model_info(short_key)
                    if info is None:
                        return
                    dataset_ui_label = "Rigplay" if HF_MODE else info.dataset_ui_label
                    self._dataset.value = dataset_ui_label
                    self._skeleton.options = get_allowed_skeleton_labels(dataset_ui_label)
                    skeleton_label = get_skeleton_display_name(info.skeleton)
                    if skeleton_label not in self._skeleton.options and self._skeleton.options:
                        skeleton_label = self._skeleton.options[0]
                    self._skeleton.value = skeleton_label
                    skeleton_key = get_skeleton_key_from_display_name(skeleton_label)
                    if skeleton_key is None:
                        return
                    models = get_models_for_dataset_skeleton(dataset_ui_label, skeleton_key, family="Kimodo")
                    self._version.options = [m.display_name for m in models]
                    self._version.value = (
                        info.display_name if info.display_name in self._version.options else self._version.options[0]
                    )
                    self._version.visible = len(models) > 1
                    self._display.content = f"**Model:** {self._version.value}"

            gui_model_selector = ModelSelectorHandle()

        with client.gui.add_folder("Examples", expand_by_default=True):
            examples_base_dir = demo.get_examples_base_dir(model_name, absolute=True)
            example_dict = viser_utils.load_example_cases(examples_base_dir)
            example_names = list(example_dict.keys())
            if not example_names:
                example_names = ["<no examples>"]
            gui_examples_dropdown = client.gui.add_dropdown(
                "Example",
                options=example_names,
                initial_value=example_names[0],
            )
            gui_load_example_button = client.gui.add_button(
                "Load Example",
                hint="Load the selected example.",
                disabled=not example_dict,
            )

            def update_examples_dropdown(
                new_example_dict: dict[str, str],
                keep_selection: bool = True,
            ) -> None:
                if not new_example_dict:
                    gui_examples_dropdown.options = ["<no examples>"]
                    gui_examples_dropdown.value = "<no examples>"
                    gui_load_example_button.disabled = True
                    return
                gui_load_example_button.disabled = False
                example_names_local = list(new_example_dict.keys())
                gui_examples_dropdown.options = example_names_local
                if keep_selection and gui_examples_dropdown.value in example_names_local:
                    return
                gui_examples_dropdown.value = example_names_local[0]

        with client.gui.add_folder("Generate", expand_by_default=True):
            gui_duration = client.gui.add_markdown(content=f"Total duration: {DEFAULT_CUR_DURATION:.1f} (sec)")

            def update_duration_gui(duration):
                gui_duration.content = f"Total duration: {duration:.1f} (sec)"

            def compute_prompt_num_frames(prompt_values):
                """Convert timeline prompt bounds to per-prompt frame counts.

                Convention in this demo:
                - All prompts except the last are treated as [start_frame, end_frame)
                  (end is exclusive).
                - The last prompt is treated as [start_frame, end_frame] (end is inclusive).
                - This assumes the prompts values are sorted by start_frame.
                """
                if len(prompt_values) == 0:
                    return []
                num_frames = []
                for i, x in enumerate(prompt_values):
                    cur = x.end_frame - x.start_frame
                    if i == len(prompt_values) - 1:
                        cur += 1
                    num_frames.append(cur)
                return num_frames

            def update_duration_auto():
                session = demo.client_sessions[client_id]
                prompt_values = sorted(
                    [x for x in timeline._prompts.values()],
                    key=lambda x: x.start_frame,
                )
                num_frames = compute_prompt_num_frames(prompt_values)
                total_nb_frames = sum(num_frames)
                cur_duration = total_nb_frames / session.model_fps
                set_new_duration(client_id, cur_duration)
                update_duration_gui(cur_duration)

            gui_num_samples_slider = client.gui.add_slider(
                "Num Samples",
                min=1,
                max=10,
                step=1,
                initial_value=1,
                visible=not HF_MODE,
            )

            gui_use_soma_layer_checkbox = client.gui.add_checkbox(
                "SOMA layer",
                initial_value=False,
                visible="soma" in (model_name or ""),
            )

            with client.gui.add_folder("Model Parameters", expand_by_default=False):
                gui_seed = client.gui.add_number("Seed", initial_value=42)

                with client.gui.add_folder("Diffusion", expand_by_default=False):
                    gui_diffusion_steps_slider = client.gui.add_slider(
                        "Denoising Steps",
                        min=2,
                        max=1000,
                        step=10,
                        initial_value=100,
                    )
                with client.gui.add_folder("Classifier-Free Guidance", expand_by_default=False):
                    gui_cfg_checkbox = client.gui.add_checkbox(
                        "Enable",
                        initial_value=True,
                        visible=True,
                    )

                    gui_cfg_text_weight_slider = client.gui.add_slider(
                        "Text Weight",
                        min=0.0,
                        max=5.0,
                        step=0.1,
                        initial_value=2.0,
                        visible=True,
                    )
                    gui_cfg_constraint_weight_slider = client.gui.add_slider(
                        "Constraint Weight",
                        min=0.0,
                        max=5.0,
                        step=0.1,
                        initial_value=2.0,
                        visible=True,
                    )
                with client.gui.add_folder(
                    "Transitions",
                    expand_by_default=False,
                    visible=SHOW_TRANSITION_PARAMS,
                ):
                    gui_num_transition_frames_slider = client.gui.add_slider(
                        "Transition frames",
                        min=1,
                        max=10,
                        step=1,
                        initial_value=NB_TRANSITION_FRAMES,
                        visible=True,
                    )
                    gui_share_transition_checkbox = client.gui.add_checkbox(  # noqa
                        "Override previous frames",
                        initial_value=False,
                        visible=True,
                    )
                    gui_percentage_transition_sharing_slider = client.gui.add_slider(
                        "Percentage overriding frames",
                        min=0,
                        max=30,
                        step=1,
                        initial_value=10,
                        visible=True,
                    )

                    @gui_share_transition_checkbox.on_update
                    def _(event: viser.GuiEvent) -> None:
                        if get_active_session(event.client) is None:
                            return
                        # disable the slider if sharing transition is False
                        gui_percentage_transition_sharing_slider.visible = gui_share_transition_checkbox.value

            with client.gui.add_folder("Post Processing", expand_by_default=False):
                _model_name = model_name or ""
                _postprocess_visible = "g1" not in _model_name
                gui_postprocess_checkbox = client.gui.add_checkbox(
                    "Enable",
                    initial_value=INIT_POSTPROCESSING,
                    hint="Apply motion post-processing (not available for G1)",
                    visible=_postprocess_visible,
                )
                gui_root_margin = client.gui.add_number(
                    "Root Margin",
                    min=0.0,
                    # max=0.5,
                    step=0.01,
                    initial_value=0.04,
                    hint="Margin for root position (meters). Lower values pin root closer to target.",
                    visible=INIT_POSTPROCESSING and _postprocess_visible,
                )

                @gui_postprocess_checkbox.on_update
                def _(event: viser.GuiEvent) -> None:
                    if get_active_session(event.client) is None:
                        return
                    # disable the slider if sharing transition is False
                    gui_root_margin.visible = gui_postprocess_checkbox.value

                gui_real_robot_rotations_checkbox = client.gui.add_checkbox(
                    "Real robot rotations",
                    initial_value=False,
                    hint="Project joint rotations to G1 real robot DoF (1-DoF per joint) and clamp to axis limits from the MuJoCo XML.",
                    visible="g1" in _model_name,
                )

            gui_generate_button = client.gui.add_button("Generate", color="green")
        with client.gui.add_folder("Constraints", expand_by_default=False):
            gui_gizmo_space_dropdown = client.gui.add_dropdown(
                "Gizmo space",
                ("Local", "World"),
                initial_value="Local",
                visible="g1" not in _model_name,
            )
            gui_edit_constraint_button = client.gui.add_button("Enter Editing Mode")
            gui_snap_to_constraint_button = client.gui.add_button(
                "Snap to Constraint",
                disabled=True,
            )
            gui_reset_constraint_button = client.gui.add_button(
                "Reset Constraint",
                disabled=True,
            )
            gui_undo_drag_button = client.gui.add_button(
                "Undo Move",
                disabled=True,
            )

            with client.gui.add_folder("Root 2D Options", expand_by_default=True):
                gui_dense_path_checkbox = client.gui.add_checkbox(
                    "Make Smooth Path",
                    initial_value=False,
                    visible=True,
                )

            gui_show_only_current_constraint_checkbox = client.gui.add_checkbox(
                "Show only Current",
                initial_value=False,
                hint="Show only constraint overlays at the current frame; uncheck to show all.",
            )

            def apply_constraint_overlay_visibility(session: ClientSession) -> None:
                demo._apply_constraint_overlay_visibility(session)

            @gui_show_only_current_constraint_checkbox.on_update
            def _(event: viser.GuiEvent) -> None:
                session = get_active_session(event.client)
                if session is None:
                    return
                session.show_only_current_constraint = gui_show_only_current_constraint_checkbox.value
                apply_constraint_overlay_visibility(session)

            gui_clear_all_constraints_button = client.gui.add_button(
                "Clear All Constraints",
                color="red",
            )

            def has_constraint_at_frame(session: ClientSession, frame_idx: int) -> bool:
                for constraint_name in ["Full-Body", "End-Effectors", "2D Root"]:
                    constraint = session.constraints.get(constraint_name)
                    if constraint is None:
                        continue
                    if frame_idx in constraint.keyframes:
                        return True
                return False

            def update_snap_to_constraint_button(session: ClientSession) -> None:
                gui_snap_to_constraint_button.disabled = not has_constraint_at_frame(session, session.frame_idx)

            def ensure_edit_snapshot(session: ClientSession, motion, frame_idx: int) -> None:
                if session.edit_mode_snapshot is None:
                    session.edit_mode_snapshot = {}
                if frame_idx in session.edit_mode_snapshot:
                    return
                session.edit_mode_snapshot[frame_idx] = {
                    "joints_pos": motion.get_joints_pos(frame_idx),
                    "joints_rot": motion.get_joints_rot(frame_idx),
                }

            def _update_dense_path(motion, session):
                constraint_info = session.constraints["2D Root"].get_constraint_info()

                if len(constraint_info["frame_idx"]) > 0:
                    min_root_frame = min(constraint_info["frame_idx"])
                    max_root_frame = max(constraint_info["frame_idx"])
                    motion.set_projected_root_pos_path(
                        constraint_info["root_pos"][:, [0, 2]],
                        min_frame_idx=min_root_frame,
                        max_frame_idx=max_root_frame,
                    )

            # Delay (ms) after last keyframe/interval move before updating path = "on release".
            DENSE_PATH_AFTER_RELEASE_MS = 300

            def _schedule_dense_path_after_release(session):
                """Schedule a single path update to run after user stops dragging."""
                if "2D Root" not in session.constraints or not session.constraints["2D Root"].dense_path:
                    return
                tdata = session.timeline_data
                if tdata.get("dense_path_after_release_timer"):
                    tdata["dense_path_after_release_timer"].cancel()
                delay = DENSE_PATH_AFTER_RELEASE_MS / 1000.0

                def run():
                    if not demo.client_active(client_id):
                        return
                    sess = demo.client_sessions[client_id]
                    tdata["dense_path_after_release_timer"] = None
                    if "2D Root" not in sess.constraints or not sess.constraints["2D Root"].dense_path:
                        return
                    mot = list(sess.motions.values())[0]
                    _update_dense_path(mot, sess)

                t = threading.Timer(delay, run)
                tdata["dense_path_after_release_timer"] = t
                t.start()

            @gui_dense_path_checkbox.on_update
            def _(event: viser.GuiEvent) -> None:
                session = get_active_session(event.client)
                if session is None:
                    return

                if gui_dense_path_checkbox.value:
                    # Make sure 0 and max_frame_idx keyframes are added to the constraint
                    # since dense path should cover full duration for best model performance
                    root_2d_track = session.timeline_data["tracks_ids"]["2D Root"]

                    # add a locked keyframe at 0
                    start_keyframe_id = client.timeline.add_locked_keyframe(  # noqa
                        root_2d_track,
                        0,
                        opacity=0.0,
                    )
                    session.timeline_data["keyframes"][start_keyframe_id] = {
                        "frame": 0,
                        "track_id": root_2d_track,
                        "locked": True,
                        "opacity": 0.0,
                        "value": None,
                    }
                    add_constraint_callback(
                        start_keyframe_id,
                        "2D Root",
                        (0, 0),
                        verbose=False,
                    )

                    # add a locked keyframe at max_frame_idx
                    end_keyframe_id = client.timeline.add_locked_keyframe(
                        root_2d_track,
                        session.max_frame_idx,
                        opacity=0.0,
                    )
                    session.timeline_data["keyframes"][end_keyframe_id] = {
                        "frame": session.max_frame_idx,
                        "track_id": root_2d_track,
                        "locked": True,
                        "opacity": 0.0,
                        "value": None,
                    }
                    add_constraint_callback(
                        end_keyframe_id,
                        "2D Root",
                        (session.max_frame_idx, session.max_frame_idx),
                        verbose=False,
                    )

                    # add a locked interval only for visual purposes
                    locked_interval = client.timeline.add_locked_interval(  # noqa
                        root_2d_track,
                        start_frame=0,
                        end_frame=session.max_frame_idx,
                    )
                    session.timeline_data["intervals"][locked_interval] = {
                        "track_id": root_2d_track,
                        "start_frame_idx": 0,
                        "end_frame_idx": session.max_frame_idx,
                        "locked": True,
                        "opacity": 0.3,
                        "value": None,
                    }

                session.constraints["2D Root"].set_dense_path(gui_dense_path_checkbox.value)
                if session.constraints["2D Root"].dense_path:
                    # update the character motion to reflect the full path
                    # will be full length by construction, no need to specify min/max frame idx
                    motion = list(session.motions.values())[0]
                    _update_dense_path(motion, session)

                # remove locked interval and locked keyframes
                if not gui_dense_path_checkbox.value:
                    # Get all locked keyframes
                    keyframes_to_remove = []
                    for uuid, keyframe in client.timeline._keyframes.items():
                        if keyframe.locked:
                            keyframes_to_remove.append(uuid)
                            _data = session.timeline_data["keyframes"][uuid]
                            remove_constraint_callback(
                                uuid,
                                constraint_type=session.timeline_data["tracks"][_data["track_id"]]["name"],
                                frame_range=(_data["frame"], _data["frame"]),
                                verbose=False,
                            )

                    intervals_to_remove = []
                    # remove all locked intervals
                    for uuid, interval in client.timeline._intervals.items():
                        if interval.locked:
                            intervals_to_remove.append(uuid)

                    # removing keyframes and intervals
                    for uuid in keyframes_to_remove:
                        client.timeline.remove_keyframe(uuid)

                    for uuid in intervals_to_remove:
                        client.timeline.remove_interval(uuid)

                apply_constraint_overlay_visibility(session)

        with client.gui.add_folder(
            "Load/Save",
            expand_by_default=False,
            visible=not HF_MODE,
        ):
            with client.gui.add_folder("Motion", expand_by_default=False):
                gui_save_motion_path_text = client.gui.add_text("Save Path", initial_value="output.npz")
                gui_save_motion_button = client.gui.add_button(
                    "Save Motion",
                    hint="Save the current motion",
                )
                gui_load_motion_path_text = client.gui.add_text("Load Path", initial_value="output.npz")
                gui_load_motion_button = client.gui.add_button(
                    "Load Motion",
                    hint="Load the selected motion",
                )
            with client.gui.add_folder("Constraints", expand_by_default=False):
                gui_save_constraints_path_text = client.gui.add_text(
                    "Save Path", initial_value="output_constraints.json"
                )
                gui_save_constraints_button = client.gui.add_button("Save Constraints")
                gui_load_constraints_path_text = client.gui.add_text(
                    "Load Path", initial_value="output_constraints.json"
                )
                gui_load_constraints_button = client.gui.add_button("Load Constraints")
            with client.gui.add_folder("Example", expand_by_default=False):
                gui_save_example_path_text = client.gui.add_text(
                    "Save Dir",
                    initial_value=os.path.join(
                        demo.get_examples_base_dir(model_name, absolute=True),
                        "custom_example_1",
                    ),
                )
                gui_save_example_button = client.gui.add_button("Save Example")
                gui_load_example_path_text = client.gui.add_text(
                    "Load Dir",
                    initial_value=os.path.join(
                        demo.get_examples_base_dir(model_name, absolute=True),
                        "custom_example_1",
                    ),
                )
                gui_load_gt_checkbox = client.gui.add_checkbox(
                    "Load GT instead",
                    initial_value=False,
                )
                gui_load_example_from_path_button = client.gui.add_button("Load Example")

            def _to_unbatched(arr: np.ndarray) -> np.ndarray:
                """Remove leading batch dimension so saved/exported files are always unbatched [T, ...]."""
                arr = np.asarray(arr)
                if arr.ndim == 4 or arr.ndim == 5:
                    return arr[0]
                if arr.ndim == 3 and arr.shape[0] == 1:
                    return arr[0]
                return arr

            def save_motion(client, save_path):
                session = demo.client_sessions[client.client_id]
                # only save the first motion
                motion = list(session.motions.values())[0]
                motion_data = {
                    "posed_joints": _to_unbatched(motion.joints_pos.cpu().numpy()),
                    "global_rot_mats": _to_unbatched(motion.joints_rot.cpu().numpy()),
                    "local_rot_mats": _to_unbatched(motion.joints_local_rot.cpu().numpy()),
                }
                if motion.foot_contacts is not None:
                    motion_data["foot_contacts"] = _to_unbatched(motion.foot_contacts.cpu().numpy())
                np.savez(save_path, **motion_data)

            @gui_save_motion_button.on_click
            def _(event: viser.GuiEvent) -> None:
                event_client = event.client
                if get_active_session(event_client) is None:
                    return

                save_path = gui_save_motion_path_text.value
                try:
                    save_motion(event_client, save_path)
                    event_client.add_notification(
                        title="Motion saved!",
                        body=f"Saved motion to {save_path}",
                        auto_close_seconds=5.0,
                        color="green",
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    event_client.add_notification(
                        title="Failed to save motion!",
                        body=str(e),
                        auto_close_seconds=5.0,
                        color="red",
                    )

            def load_motion(client, load_path):
                session = demo.client_sessions[client.client_id]

                # Load the NPZ file
                data = np.load(load_path)

                # Extract motion data - handle different possible key names
                if "joints_pos" in data:
                    joints_pos = torch.from_numpy(data["joints_pos"]).to(demo.device)
                elif "posed_joints" in data:
                    joints_pos = torch.from_numpy(data["posed_joints"]).to(demo.device)
                else:
                    raise ValueError("NPZ file must contain 'joints_pos' or 'posed_joints'")

                if "joints_rot" in data:
                    joints_rot = torch.from_numpy(data["joints_rot"]).to(demo.device)
                elif "global_rot_mats" in data:
                    joints_rot = torch.from_numpy(data["global_rot_mats"]).to(demo.device)
                else:
                    raise ValueError("NPZ file must contain 'joints_rot' or 'global_rot_mats'")

                # Foot contacts are optional
                foot_contacts = None
                if "foot_contacts" in data:
                    foot_contacts = torch.from_numpy(data["foot_contacts"]).to(demo.device)

                # Support both batched [B, T, J, 3] and unbatched [T, J, 3]; take first sample if batched
                if joints_pos.ndim == 4:
                    joints_pos = joints_pos[0]
                if joints_rot.ndim == 5:
                    joints_rot = joints_rot[0]
                if foot_contacts is not None and foot_contacts.ndim == 3:
                    foot_contacts = foot_contacts[0]

                # Motion must have the same number of joints as the current model's skeleton
                num_joints_loaded = joints_pos.shape[1]
                num_joints_skeleton = session.skeleton.nbjoints
                if num_joints_loaded != num_joints_skeleton:
                    raise ValueError(
                        f"The loaded motion has {num_joints_loaded} joints but the current model "
                        f"({session.model_name}) has {num_joints_skeleton} joints. "
                        "Load a motion generated with the same skeleton, or switch the model to match the motion."
                    )
                if joints_rot.shape[1] != num_joints_skeleton:
                    raise ValueError(
                        f"Rotation data has {joints_rot.shape[1]} joints but the current model has "
                        f"{num_joints_skeleton} joints. The NPZ may be corrupted or from a different skeleton."
                    )

                # Apply G1 real robot projection (1-DoF per joint + axis limits) if enabled.
                if (
                    "g1" in session.model_name
                    and isinstance(session.skeleton, G1Skeleton34)
                    and gui_real_robot_rotations_checkbox.value
                ):
                    joints_pos, joints_rot = generation.apply_g1_real_robot_projection(
                        session.skeleton, joints_pos, joints_rot
                    )

                # Update duration and frame range based on loaded motion
                num_frames = joints_pos.shape[0]
                duration = num_frames / session.model_fps

                # Update GUI elements
                session.cur_duration = duration
                session.max_frame_idx = num_frames - 1

                # Clear existing motions and add the loaded one
                demo.clear_motions(client.client_id)
                demo.add_character_motion(
                    client,
                    session.skeleton,
                    joints_pos,
                    joints_rot,
                    foot_contacts,
                )

                # Reset to frame 0
                demo.set_frame(client.client_id, 0)

            @gui_load_motion_button.on_click
            def _(event: viser.GuiEvent) -> None:
                event_client = event.client
                session = get_active_session(event_client)
                if session is None:
                    return

                try:
                    load_path = gui_load_motion_path_text.value
                    load_motion(event_client, load_path)

                    event_client.add_notification(
                        title="Motion loaded!",
                        body=f"Loaded motion from {load_path} ({session.max_frame_idx + 1} frames, {session.cur_duration:.2f}s)",
                        auto_close_seconds=5.0,
                        color="green",
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    event_client.add_notification(
                        title="Failed to load motion!",
                        body=str(e),
                        auto_close_seconds=10.0,
                        color="red",
                    )

            def save_constraints(client, save_path):
                session = demo.client_sessions[client.client_id]
                # Keep save behavior aligned with demo frame convention:
                # valid frame indices are [0, max_frame_idx], so count is +1.
                num_frames = session.max_frame_idx + 1
                constraints_lst = demo.compute_model_constraints_lst(session, num_frames)
                save_constraints_lst(save_path, constraints_lst)

            @gui_save_constraints_button.on_click
            def _(event: viser.GuiEvent) -> None:
                event_client = event.client
                if get_active_session(event_client) is None:
                    return

                try:
                    save_path = gui_save_constraints_path_text.value
                    save_constraints(event_client, save_path)
                    event_client.add_notification(
                        title="Constraints saved!",
                        body=f"Saved constraints to {save_path}",
                        auto_close_seconds=5.0,
                        color="green",
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    event_client.add_notification(
                        title="Failed to save constraints!",
                        body=str(e),
                        auto_close_seconds=10.0,
                        color="red",
                    )

            def load_constraints(client, load_path):
                session = demo.client_sessions[client.client_id]
                constraints_lst = load_constraints_lst(load_path, skeleton=session.skeleton)

                # Clear existing constraints first
                with session.timeline_data["keyframe_update_lock"]:
                    for constraint in list(session.constraints.values()):
                        constraint.clear()
                    client.timeline.clear_keyframes()
                    client.timeline.clear_intervals()

                # Add loaded constraints to the session
                # We need to directly add constraint data, not read from current motion
                device = demo.device
                for constraint_obj in constraints_lst:
                    constraint_type = constraint_obj.name

                    # decompose the frame indices into intervals or single keyframes
                    frame_indices = constraint_obj.frame_indices
                    (
                        intervals,
                        intervals_indices,
                        single_frames,
                        single_frames_indices,
                    ) = extract_intervals_and_singles(frame_indices)

                    load_targets: list[dict] = []
                    root_pos = None

                    if constraint_type == "root2d":
                        # smooth_root_2d is [T, 2] (x, z), convert to [T, 3] (x, 0, z)
                        num_frames = constraint_obj.smooth_root_2d.shape[0]
                        root_pos = torch.zeros(num_frames, 3, device=device)
                        root_pos[:, 0] = constraint_obj.smooth_root_2d[:, 0]
                        root_pos[:, 2] = constraint_obj.smooth_root_2d[:, 1]
                        load_targets = [
                            {
                                "track_name": "2D Root",
                                "constraint_track": session.constraints["2D Root"],
                            }
                        ]
                    elif constraint_type == "fullbody":
                        load_targets = [
                            {
                                "track_name": "Full-Body",
                                "constraint_track": session.constraints["Full-Body"],
                            }
                        ]
                    elif constraint_type in {
                        "left-hand",
                        "right-hand",
                        "left-foot",
                        "right-foot",
                    }:
                        track_name = {
                            "left-hand": "Left Hand",
                            "right-hand": "Right Hand",
                            "left-foot": "Left Foot",
                            "right-foot": "Right Foot",
                        }[constraint_type]
                        load_targets = [
                            {
                                "track_name": track_name,
                                "constraint_track": session.constraints["End-Effectors"],
                                "joint_names": constraint_obj.joint_names,
                                "end_effector_type": constraint_type,
                            }
                        ]
                    elif constraint_type in {"end-effector", "end-effectors"}:
                        # Backward-compatible loader:
                        # split a generic end-effector constraint into per-limb timeline tracks.
                        joint_names_set = set(constraint_obj.joint_names)
                        for jname, track_name, eff_type in [
                            ("LeftHand", "Left Hand", "left-hand"),
                            ("RightHand", "Right Hand", "right-hand"),
                            ("LeftFoot", "Left Foot", "left-foot"),
                            ("RightFoot", "Right Foot", "right-foot"),
                        ]:
                            if jname not in joint_names_set:
                                continue
                            target_joint_names = [jname]
                            if "Hips" in joint_names_set:
                                target_joint_names.append("Hips")
                            load_targets.append(
                                {
                                    "track_name": track_name,
                                    "constraint_track": session.constraints["End-Effectors"],
                                    "joint_names": target_joint_names,
                                    "end_effector_type": eff_type,
                                }
                            )
                        if not load_targets:
                            raise KeyError(
                                "No recognized end-effector joint in constraint "
                                f"joint_names={constraint_obj.joint_names}"
                            )
                    else:
                        raise KeyError(f"Unsupported constraint type in loader: {constraint_type}")

                    for target in load_targets:
                        track_id = session.timeline_data["tracks_ids"][target["track_name"]]
                        constraint_track = target["constraint_track"]

                        # add intervals
                        for (start_idx, end_idx), (start_idx_t, end_idx_t) in zip(intervals, intervals_indices):
                            # Add to timeline
                            interval_id = client.timeline.add_interval(track_id, start_idx, end_idx)
                            session.timeline_data["intervals"][interval_id] = {
                                "track_id": track_id,
                                "start_frame_idx": start_idx,
                                "end_frame_idx": end_idx,
                                "locked": False,
                                "opacity": 1.0,
                                "value": None,
                            }
                            if constraint_type == "root2d":
                                constraint_track.add_interval(
                                    interval_id,
                                    start_idx,
                                    end_idx,
                                    root_pos[start_idx_t : end_idx_t + 1],
                                )
                            elif constraint_type == "fullbody":
                                constraint_track.add_interval(
                                    interval_id,
                                    start_idx,
                                    end_idx,
                                    constraint_obj.global_joints_positions[start_idx_t : end_idx_t + 1],
                                    constraint_obj.global_joints_rots[start_idx_t : end_idx_t + 1],
                                )
                            else:
                                constraint_track.add_interval(
                                    interval_id,
                                    start_idx,
                                    end_idx,
                                    constraint_obj.global_joints_positions[start_idx_t : end_idx_t + 1],
                                    constraint_obj.global_joints_rots[start_idx_t : end_idx_t + 1],
                                    target["joint_names"],
                                    target["end_effector_type"],
                                )

                        # add keyframes
                        for frame, frame_t in zip(single_frames, single_frames_indices):
                            # Add to timeline
                            keyframe_id = client.timeline.add_keyframe(track_id, frame)
                            session.timeline_data["keyframes"][keyframe_id] = {
                                "track_id": track_id,
                                "frame": frame,
                                "locked": False,
                                "opacity": 1.0,
                                "value": None,
                            }
                            if constraint_type == "root2d":
                                constraint_track.add_keyframe(
                                    keyframe_id,
                                    frame,
                                    root_pos[frame_t],
                                )
                            elif constraint_type == "fullbody":
                                constraint_track.add_keyframe(
                                    keyframe_id,
                                    frame,
                                    constraint_obj.global_joints_positions[frame_t],
                                    constraint_obj.global_joints_rots[frame_t],
                                )
                            else:
                                constraint_track.add_keyframe(
                                    keyframe_id,
                                    frame,
                                    constraint_obj.global_joints_positions[frame_t],
                                    constraint_obj.global_joints_rots[frame_t],
                                    target["joint_names"],
                                    target["end_effector_type"],
                                )

            @gui_load_constraints_button.on_click
            def _(event: viser.GuiEvent) -> None:
                event_client = event.client
                if get_active_session(event_client) is None:
                    return

                try:
                    load_path = gui_load_constraints_path_text.value
                    load_constraints(event_client, load_path)
                    session = demo.client_sessions[event_client.client_id]
                    apply_constraint_overlay_visibility(session)

                    event_client.add_notification(
                        title="Constraints loaded!",
                        body=f"Loaded constraints from {load_path}",
                        auto_close_seconds=5.0,
                        color="green",
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    event_client.add_notification(
                        title="Failed to load constraints!",
                        body=str(e),
                        auto_close_seconds=10.0,
                        color="red",
                    )

        with client.gui.add_folder("Exports", expand_by_default=False):
            with client.gui.add_folder("Screenshot", expand_by_default=False, visible=not HF_MODE):
                gui_screenshot_path_text = client.gui.add_text(
                    "Save Path",
                    initial_value="render.png",
                    hint="Filename for the screenshot (PNG).",
                )
                gui_screenshot_button = client.gui.add_button(
                    "Download Screenshot",
                    hint="Capture the current canvas and download a PNG.",
                )
            with client.gui.add_folder("Video", expand_by_default=False, visible=not HF_MODE):
                gui_video_path_text = client.gui.add_text(
                    "Save Path",
                    initial_value="render.mp4",
                    hint="Filename for the video (MP4).",
                )
                gui_video_button = client.gui.add_button(
                    "Download Video",
                    hint="Render every frame and download as MP4.",
                )
            with client.gui.add_folder("Motion", expand_by_default=True):
                gui_download_name_text = client.gui.add_text(
                    "Name",
                    initial_value="output",
                    hint="Base filename to save as (extension will be added based on format if omitted).",
                )
                gui_download_format_dropdown = client.gui.add_dropdown(
                    "Format",
                    options=_get_download_format_options(is_g1="g1" in model_name),
                    initial_value="NPZ",
                )
                gui_download_button = client.gui.add_button(
                    "Download",
                    hint="Download the current motion (format + name above).",
                )

            def _download_bytes_to_browser(
                event_client: viser.ClientHandle,
                *,
                data: bytes,
                filename: str,
                mime_type: str = "application/octet-stream",
            ) -> None:
                """Trigger a browser download for an in-memory byte payload.

                Important: this intentionally does NOT use `showSaveFilePicker()` to avoid
                Chrome/Edge's file-write permission prompt ("this site can see edits you make").
                If you want "always ask where to save", configure your browser download settings.
                """
                import base64
                import json

                # Base64 is the most robust way to move binary over our websocket JS channel.
                b64 = base64.b64encode(data).decode("ascii")
                js = f"""
(() => {{
  const filename = {json.dumps(filename)};
  const mimeType = {json.dumps(mime_type)};
  const b64 = {json.dumps(b64)};

  // Decode base64 -> Uint8Array.
  const binStr = atob(b64);
  const bytes = new Uint8Array(binStr.length);
  for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
  const blob = new Blob([bytes], {{ type: mimeType }});

  // Standard browser download behavior.
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}})();
"""
                # Reuse viser’s JS execution mechanism (used for Plotly setup).
                from viser import _messages as _viser_messages

                event_client.gui._websock_interface.queue_message(  # type: ignore[attr-defined]
                    _viser_messages.RunJavascriptMessage(source=js)
                )

            def _motion_to_npz_bytes(motion) -> bytes:
                import io

                motion_data = {
                    "posed_joints": _to_unbatched(motion.joints_pos.detach().cpu().numpy()),
                    "global_rot_mats": _to_unbatched(motion.joints_rot.detach().cpu().numpy()),
                    "local_rot_mats": _to_unbatched(motion.joints_local_rot.detach().cpu().numpy()),
                }
                if getattr(motion, "foot_contacts", None) is not None:
                    motion_data["foot_contacts"] = _to_unbatched(motion.foot_contacts.detach().cpu().numpy())
                buf = io.BytesIO()
                np.savez(buf, **motion_data)
                return buf.getvalue()

            def _motion_to_csv_bytes(motion, skeleton) -> bytes:
                import io

                from kimodo.exports.mujoco import MujocoQposConverter

                if not isinstance(skeleton, G1Skeleton34):
                    raise ValueError("CSV export is only supported for the G1 skeleton.")

                converter = MujocoQposConverter(skeleton)
                qpos = converter.dict_to_qpos(
                    {
                        "local_rot_mats": motion.joints_local_rot.detach().cpu(),
                        "root_positions": motion.joints_pos[:, skeleton.root_idx, :].detach().cpu(),
                    },
                    numpy=True,
                )

                buf = io.StringIO()
                np.savetxt(buf, qpos, delimiter=",")
                return buf.getvalue().encode("utf-8")

            def _coerce_download_filename(raw_name: str, *, ext: str) -> str:
                """Coerce a user-entered filename to a safe basename with the desired extension.

                - If empty: uses "output{ext}"
                - If no extension: appends ext
                - If endswith a known export extension: rewrites extension to ext (prevents mismatches)
                - Any provided directory components are stripped
                """
                import os

                name = (raw_name or "").strip()
                name = os.path.basename(name.replace("\\", "/"))
                if name == "":
                    return f"output{ext}"

                known_exts = (".npz", ".bvh", ".csv", ".png", ".mp4")
                lower = name.lower()
                if lower.endswith(known_exts):
                    return os.path.splitext(name)[0] + ext

                root, cur_ext = os.path.splitext(name)
                if cur_ext == "":
                    return name + ext
                return name

            def _get_render_size(event_client: viser.ClientHandle) -> tuple[int, int]:
                width = int(event_client.camera.image_width)
                height = int(event_client.camera.image_height)
                if width <= 0 or height <= 0:
                    # Fall back to a reasonable default if the camera hasn't synced yet.
                    return (1280, 720)
                return (width, height)

            def _round_up_to_multiple(value: int, multiple: int) -> int:
                if multiple <= 0:
                    return value
                return ((value + multiple - 1) // multiple) * multiple

            def _download_canvas_to_browser(event_client: viser.ClientHandle, *, filename: str) -> None:
                """Use the client-side canvas save path to avoid server-side renders."""
                import json

                js = f"""
(() => {{
  const filename = {json.dumps(filename)};
  const canvases = Array.from(document.querySelectorAll("canvas"));
  if (!canvases.length) {{
    console.error("No canvases found to save.");
    return;
  }}
  // Pick the largest canvas by area (usually the main 3D view).
  const canvas = canvases.reduce((best, cur) => {{
    const bestArea = (best?.width || 0) * (best?.height || 0);
    const curArea = (cur?.width || 0) * (cur?.height || 0);
    return curArea > bestArea ? cur : best;
  }}, null);
  if (!canvas) {{
    console.error("No canvas selected to save.");
    return;
  }}
  canvas.toBlob((blob) => {{
    if (!blob) {{
      console.error("Export failed");
      return;
    }}
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }}, "image/png");
}})();
"""
                from viser import _messages as _viser_messages

                event_client.gui._websock_interface.queue_message(  # type: ignore[attr-defined]
                    _viser_messages.RunJavascriptMessage(source=js)
                )

            @gui_screenshot_button.on_click
            def _(event: viser.GuiEvent) -> None:
                event_client = event.client
                if get_active_session(event_client) is None:
                    return

                try:
                    filename = _coerce_download_filename(
                        str(gui_screenshot_path_text.value),
                        ext=".png",
                    )
                    _download_canvas_to_browser(event_client, filename=filename)
                    event_client.add_notification(
                        title="Screenshot download started",
                        body=f"Saving {filename}",
                        auto_close_seconds=5.0,
                        color="green",
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    event_client.add_notification(
                        title="Failed to download screenshot!",
                        body=str(e),
                        auto_close_seconds=10.0,
                        color="red",
                    )

            @gui_video_button.on_click
            def _(event: viser.GuiEvent) -> None:
                event_client = event.client
                session = get_active_session(event_client)
                if session is None:
                    return
                recording_notification: viser.NotificationHandle | None = None
                try:
                    recording_notification = event_client.add_notification(
                        title="Recording video...",
                        body="Saving frames, please wait.",
                        loading=True,
                        with_close_button=False,
                        auto_close_seconds=None,
                        color="blue",
                    )
                    event_client.timeline.disable_constraints()
                    width, height = _get_render_size(event_client)
                    # Avoid ffmpeg macro block resizing warnings.
                    width = _round_up_to_multiple(width, 16)
                    height = _round_up_to_multiple(height, 16)
                    original_frame = session.frame_idx
                    frames = []
                    for frame_idx in range(session.max_frame_idx + 1):
                        demo.set_frame(
                            event_client.client_id,
                            frame_idx,
                            update_timeline=True,
                        )
                        frames.append(
                            event_client.get_render(
                                height=height,
                                width=width,
                                transport_format="jpeg",
                            )
                        )

                    # Restore the original frame (and timeline).
                    demo.set_frame(event_client.client_id, original_frame)

                    import imageio.v3 as iio

                    filename = _coerce_download_filename(
                        str(gui_video_path_text.value),
                        ext=".mp4",
                    )
                    payload = iio.imwrite(
                        "<bytes>",
                        frames,
                        extension=".mp4",
                        fps=float(session.model_fps),
                        codec="h264",
                        plugin="pyav",
                    )
                    event_client.send_file_download(filename, payload, save_immediately=True)
                    event_client.add_notification(
                        title="Video download started",
                        body=f"Saving {filename}",
                        auto_close_seconds=5.0,
                        color="green",
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    event_client.add_notification(
                        title="Failed to download video!",
                        body=str(e),
                        auto_close_seconds=10.0,
                        color="red",
                    )
                finally:
                    event_client.timeline.enable_constraints()
                    if recording_notification is not None:
                        recording_notification.remove()

            @gui_download_button.on_click
            def _(event: viser.GuiEvent) -> None:
                event_client = event.client
                session = get_active_session(event_client)
                if session is None:
                    return
                motion = list(session.motions.values())[0]
                try:
                    fmt = str(gui_download_format_dropdown.value).upper()
                    raw_name = str(gui_download_name_text.value)

                    if fmt == "BVH":
                        filename = _coerce_download_filename(raw_name, ext=".bvh")
                        payload = motion_to_bvh_bytes(
                            motion.joints_local_rot,
                            motion.joints_pos[:, session.skeleton.root_idx, :],  # root positions
                            skeleton=session.skeleton,
                            fps=float(session.model_fps),
                        )
                        mime = "text/plain"
                    elif fmt == "CSV":
                        filename = _coerce_download_filename(raw_name, ext=".csv")
                        payload = _motion_to_csv_bytes(motion, session.skeleton)
                        mime = "text/csv"
                    else:
                        # Default to NPZ (most common and matches existing save/load).
                        filename = _coerce_download_filename(raw_name, ext=".npz")
                        payload = _motion_to_npz_bytes(motion)
                        mime = "application/octet-stream"

                    _download_bytes_to_browser(
                        event_client,
                        data=payload,
                        filename=filename,
                        mime_type=mime,
                    )

                    event_client.add_notification(
                        title="Download started",
                        body=f"Saving {filename}",
                        auto_close_seconds=5.0,
                        color="green",
                    )
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    event_client.add_notification(
                        title="Failed to download motion!",
                        body=str(e),
                        auto_close_seconds=10.0,
                        color="red",
                    )

        @gui_save_example_button.on_click
        def _(event: viser.GuiEvent) -> None:
            from kimodo.tools import save_json

            event_client = event.client
            session = get_active_session(event_client)
            if session is None:
                return

            save_dir = gui_save_example_path_text.value
            if os.path.exists(save_dir):
                event_client.add_notification(
                    title="Failed to save example!",
                    body="Example directory already exists",
                    auto_close_seconds=10.0,
                    color="red",
                )
                return

            try:
                os.makedirs(save_dir)
                # save the constraints
                constraint_path = os.path.join(save_dir, "constraints.json")
                save_constraints(event_client, constraint_path)
                # save the motion
                motion_path = os.path.join(save_dir, "motion.npz")
                save_motion(event_client, motion_path)
                # save the gui metadata
                meta_path = os.path.join(save_dir, "meta.json")
                prompt_texts = []
                prompt_durations_sec = []
                prompt_values = sorted(
                    [x for x in client.timeline._prompts.values()],
                    key=lambda x: x.start_frame,
                )
                for i, prompt in enumerate(prompt_values):
                    prompt_texts.append(prompt.text)
                    # Match demo/generation convention:
                    # non-last prompts: [start, end) ; last prompt: [start, end].
                    n_frames = prompt.end_frame - prompt.start_frame
                    if i == len(prompt_values) - 1:
                        n_frames += 1
                    prompt_durations_sec.append(n_frames / session.model_fps)
                if len(prompt_texts) == 1:
                    meta_info = {
                        "text": prompt_texts[0],
                        "duration": prompt_durations_sec[0],
                    }
                else:
                    meta_info = {
                        "texts": prompt_texts,
                        "durations": prompt_durations_sec,
                    }
                meta_info["num_samples"] = gui_num_samples_slider.value
                meta_info["seed"] = gui_seed.value
                meta_info["diffusion_steps"] = gui_diffusion_steps_slider.value
                meta_info["cfg"] = {
                    "enabled": gui_cfg_checkbox.value,
                    "text_weight": gui_cfg_text_weight_slider.value,
                    "constraint_weight": gui_cfg_constraint_weight_slider.value,
                }
                save_json(meta_path, meta_info)

                # update the example dropdown
                session.example_dict = viser_utils.load_example_cases(session.examples_base_dir)
                update_examples_dropdown(session.example_dict, keep_selection=True)

                event_client.add_notification(
                    title="Example saved!",
                    body=f"Saved example to {save_dir}",
                    auto_close_seconds=5.0,
                    color="green",
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                event_client.add_notification(
                    title="Failed to save example!",
                    body=str(e),
                    auto_close_seconds=10.0,
                    color="red",
                )

        def set_new_duration(client_id, new_duration):
            session = demo.client_sessions[client_id]
            session.cur_duration = new_duration
            update_duration_gui(new_duration)
            session.max_frame_idx = int(session.cur_duration * session.model_fps - 1)
            if session.frame_idx > session.max_frame_idx:
                demo.set_frame(client_id, session.max_frame_idx)

        def apply_model_selection(new_model_name: str) -> None:
            session = demo.client_sessions[client_id]
            if new_model_name == session.model_name:
                return

            session.playing = False  # Pause playback when switching models.

            old_model_fps = session.model_fps
            old_duration = session.cur_duration
            old_prompts = [
                (prompt.text, prompt.start_frame, prompt.end_frame) for prompt in client.timeline._prompts.values()
            ]
            old_default_zoom_frames = client.timeline._default_num_frames_zoom
            old_max_zoom_frames = client.timeline._max_frames_zoom

            model_bundle = demo.load_model(new_model_name)

            # Clear motions and constraints when switching models.
            if session.edit_mode and session.motions:
                exit_editing_mode(session)
            session.edit_mode = False
            demo.clear_motions(client_id)
            with session.timeline_data["keyframe_update_lock"]:
                for constraint in list(session.constraints.values()):
                    constraint.clear()
                session.constraints = demo.build_constraint_tracks(client, model_bundle.skeleton)
                session.timeline_data["keyframes"] = {}
                session.timeline_data["intervals"] = {}
                client.timeline.clear_keyframes()
                client.timeline.clear_intervals()

            session.model_name = new_model_name
            session.model_fps = model_bundle.model_fps
            session.skeleton = model_bundle.skeleton
            session.motion_rep = model_bundle.motion_rep
            session.cur_duration = old_duration
            session.max_frame_idx = int(session.cur_duration * session.model_fps - 1)
            session.frame_idx = 0
            session.edit_mode = False

            demo.set_timeline_defaults(client.timeline, session.model_fps)
            client.timeline.set_current_frame(0)
            gui_model_fps.value = session.model_fps
            update_duration_gui(session.cur_duration)

            if old_model_fps > 0:
                default_zoom_seconds = old_default_zoom_frames / old_model_fps
                max_zoom_seconds = old_max_zoom_frames / old_model_fps
                new_default_zoom = int(round(default_zoom_seconds * session.model_fps))
                new_max_zoom = int(round(max_zoom_seconds * session.model_fps))
                new_default_zoom = max(1, new_default_zoom)
                new_max_zoom = max(new_default_zoom, new_max_zoom)
                client.timeline.set_zoom_settings(
                    default_num_frames_zoom=new_default_zoom,
                    max_frames_zoom=new_max_zoom,
                )

            client.timeline.clear_prompts()
            if old_prompts and old_model_fps > 0:
                for i, (prompt_text, start_frame, end_frame) in enumerate(old_prompts):
                    start_sec = start_frame / old_model_fps
                    end_sec = end_frame / old_model_fps
                    new_start = int(round(start_sec * session.model_fps))
                    new_end = int(round(end_sec * session.model_fps))
                    new_start = max(0, min(new_start, session.max_frame_idx))
                    new_end = max(new_start, min(new_end, session.max_frame_idx))
                    color = PROMPT_COLORS[i % len(PROMPT_COLORS)]
                    client.timeline.add_prompt(prompt_text, new_start, new_end, color=color)

            session.examples_base_dir = demo.get_examples_base_dir(new_model_name, absolute=True)
            session.example_dict = viser_utils.load_example_cases(session.examples_base_dir)
            update_examples_dropdown(session.example_dict, keep_selection=False)
            gui_save_example_path_text.value = os.path.join(
                demo.get_examples_base_dir(new_model_name, absolute=True),
                "custom_example_1",
            )
            gui_load_example_path_text.value = os.path.join(
                demo.get_examples_base_dir(new_model_name, absolute=True),
                "custom_example_1",
            )

            demo.add_character_motion(client, session.skeleton)
            apply_constraint_overlay_visibility(session)

        def _update_version_and_display_from_dataset_skeleton() -> None:
            dataset_ui = gui_dataset_selector.value
            skeleton_display = gui_skeleton_selector.value
            skeleton_val = get_skeleton_key_from_display_name(skeleton_display)
            if skeleton_val is None:
                return
            models = get_models_for_dataset_skeleton(dataset_ui, skeleton_val, family="Kimodo")
            if not models:
                return
            gui_version_selector.options = [m.display_name for m in models]
            gui_version_selector.value = models[0].display_name
            gui_version_selector.visible = len(models) > 1
            gui_model_display.content = f"**Model:** {models[0].display_name}"

        def _update_visibility_for_loaded_model(loaded_model_name: str) -> None:
            """Update model-specific controls from the currently loaded model only."""
            if not loaded_model_name:
                return
            gui_use_soma_layer_checkbox.visible = "soma" in loaded_model_name
            _is_g1 = "g1" in loaded_model_name
            _update_download_format_options(gui_download_format_dropdown, is_g1=_is_g1)
            gui_real_robot_rotations_checkbox.visible = _is_g1
            gui_postprocess_checkbox.visible = not _is_g1
            gui_root_margin.visible = not _is_g1 and gui_postprocess_checkbox.value
            if _is_g1:
                gui_gizmo_space_dropdown.value = "Local"
            gui_gizmo_space_dropdown.visible = not _is_g1
            gui_gizmo_space_dropdown.disabled = _is_g1

        def _on_load_model_click(event: viser.GuiEvent) -> None:
            """Load the currently selected model (called from Load model button)."""
            if get_active_session(event.client) is None:
                return
            new_model_name = gui_model_selector.value
            if not new_model_name:
                return
            info = get_model_info(new_model_name)
            if info is None:
                return
            session = demo.client_sessions[event.client.client_id]
            if new_model_name == session.model_name:
                return
            loading_notif = event.client.add_notification(
                title="Loading model...",
                body=f"Loading {info.display_name}",
                loading=True,
                with_close_button=False,
            )
            try:
                apply_model_selection(new_model_name)
                _update_visibility_for_loaded_model(new_model_name)
                loading_notif.title = "Model loaded"
                loading_notif.body = f"{info.display_name} is ready."
                loading_notif.loading = False
                loading_notif.with_close_button = True
                loading_notif.auto_close_seconds = 5.0
                loading_notif.color = "green"
            except Exception as e:
                loading_notif.loading = False
                loading_notif.with_close_button = True
                event.client.add_notification(
                    title="Model failed to load",
                    body=str(e),
                    color="red",
                    auto_close_seconds=10.0,
                )
                gui_model_selector.set_from_short_key(session.model_name)

        @gui_load_model_button.on_click
        def _(event: viser.GuiEvent) -> None:
            _on_load_model_click(event)

        @gui_dataset_selector.on_update
        def _(event: viser.GuiEvent) -> None:
            if get_active_session(event.client) is None:
                return
            skeleton_labels = get_allowed_skeleton_labels(gui_dataset_selector.value)
            gui_skeleton_selector.options = skeleton_labels
            gui_skeleton_selector.value = skeleton_labels[0] if skeleton_labels else ""
            _update_version_and_display_from_dataset_skeleton()

        @gui_skeleton_selector.on_update
        def _(event: viser.GuiEvent) -> None:
            if get_active_session(event.client) is None:
                return
            _update_version_and_display_from_dataset_skeleton()

        @gui_version_selector.on_update
        def _(event: viser.GuiEvent) -> None:
            if get_active_session(event.client) is None:
                return
            info = get_model_info(gui_model_selector.value)
            if info is not None:
                gui_model_display.content = f"**Model:** {info.display_name}"

        @gui_use_soma_layer_checkbox.on_update
        def _(event: viser.GuiEvent) -> None:
            session = get_active_session(event.client)
            if session is None or "soma" not in (session.model_name or ""):
                return

            loading_notif = event.client.add_notification(
                title="Applying SOMA layer...",
                body="Updating mesh.",
                loading=True,
                with_close_button=False,
            )
            try:
                current_motion = list(session.motions.values())[0] if session.motions else None
                current_frame_idx = session.frame_idx

                # Recreate the character to apply the new SOMA mesh mode selection.
                demo.clear_motions(event.client.client_id)
                if current_motion is None:
                    demo.add_character_motion(event.client, session.skeleton)
                else:
                    demo.add_character_motion(
                        event.client,
                        session.skeleton,
                        current_motion.joints_pos,
                        current_motion.joints_rot,
                        current_motion.foot_contacts,
                    )

                demo.set_frame(event.client.client_id, current_frame_idx)
            except Exception as e:
                print(e)
                event.client.add_notification(
                    title="SOMA layer failed",
                    body=str(e),
                    color="red",
                    auto_close_seconds=10.0,
                )
                gui_use_soma_layer_checkbox.value = not gui_use_soma_layer_checkbox.value
            finally:
                loading_notif.loading = False
                loading_notif.with_close_button = True
                loading_notif.auto_close_seconds = 2.0

        @gui_real_robot_rotations_checkbox.on_update
        def _(event: viser.GuiEvent) -> None:
            session = get_active_session(event.client)
            if session is None or "g1" not in session.model_name:
                return
            if not isinstance(session.skeleton, G1Skeleton34) or not session.motions:
                return
            if not gui_real_robot_rotations_checkbox.value:
                return
            # Reproject all displayed G1 motions to real robot DoF (1-DoF per joint + axis limits).
            from kimodo.skeleton import global_rots_to_local_rots

            current_frame_idx = session.frame_idx
            for motion in session.motions.values():
                if motion.length <= 1:
                    continue
                rest_pos = motion.joints_pos[0:1]
                rest_rot = motion.joints_rot[0:1]
                same_as_rest = (motion.joints_pos - rest_pos).abs().max().item() < 1e-6 and (
                    motion.joints_rot - rest_rot
                ).abs().max().item() < 1e-6
                if same_as_rest:
                    continue
                new_pos, new_rot = generation.apply_g1_real_robot_projection(
                    session.skeleton,
                    motion.joints_pos,
                    motion.joints_rot,
                )
                motion.joints_pos = new_pos
                motion.joints_rot = new_rot
                motion.joints_local_rot = global_rots_to_local_rots(new_rot, session.skeleton)
                # Refresh skeleton and skinned mesh caches so the viz uses new positions.
                motion.precompute_mesh_info()
            demo.set_frame(event.client.client_id, current_frame_idx)
            event.client.add_notification(
                title="Real robot projection applied",
                body="The motion is projected to G1 real robot DoF (1-DoF per joint, clamped to axis limits).",
                auto_close_seconds=4.0,
                color="green",
            )

        def load_example_from_path(
            event_client: viser.ClientHandle,
            example_path: str,
            load_gt: bool = False,
        ) -> None:
            from kimodo.meta import parse_prompts_from_meta
            from kimodo.tools import load_json

            session = get_active_session(event_client)
            if session is None:
                return

            # Pause playback when loading an example.
            session.playing = False

            if not os.path.isdir(example_path):
                event_client.add_notification(
                    title="Example path not found",
                    body=f"Directory does not exist: {example_path}",
                    auto_close_seconds=5.0,
                    color="red",
                )
                return

            try:
                # constraints
                constraints_path = os.path.join(example_path, "constraints.json")
                if os.path.exists(constraints_path):
                    load_constraints(event_client, constraints_path)
                else:
                    # clear all existing constraints
                    with session.timeline_data["keyframe_update_lock"]:
                        for constraint in list(session.constraints.values()):
                            constraint.clear()
                        event_client.timeline.clear_keyframes()
                        event_client.timeline.clear_intervals()
                # motion
                motion_filename = "gt_motion.npz" if load_gt else "motion.npz"
                motion_path = os.path.join(example_path, motion_filename)
                if os.path.exists(motion_path):
                    load_motion(event_client, motion_path)
                # metadata
                meta_path = os.path.join(example_path, "meta.json")
                if os.path.exists(meta_path):
                    meta_info = load_json(meta_path)
                    event_client.timeline.clear_prompts()

                    texts, durations_sec = parse_prompts_from_meta(meta_info)
                    fps = session.model_fps
                    # Convert durations (seconds) to consecutive frame bounds
                    num_frames = 0
                    frame_bounds = []
                    for i, d in enumerate(durations_sec):
                        n_frames = max(1, int(round(d * fps)))
                        start_frame = num_frames
                        # Inverse of compute_prompt_num_frames():
                        # non-last prompts end at next prompt start (exclusive),
                        # last prompt includes its end frame.
                        if i == len(durations_sec) - 1:
                            end_frame = num_frames + n_frames - 1
                        else:
                            end_frame = num_frames + n_frames
                        frame_bounds.append((start_frame, end_frame))
                        num_frames += n_frames

                    # Adapt timeline zoom to the loaded motion.
                    target_visible_frames = int(math.ceil(1.10 * num_frames))
                    event_client.timeline.set_zoom_settings(
                        default_num_frames_zoom=target_visible_frames,
                    )

                    for i, (prompt_text, (start_frame, end_frame)) in enumerate(zip(texts, frame_bounds)):
                        color = PROMPT_COLORS[i % len(PROMPT_COLORS)]
                        event_client.timeline.add_prompt(prompt_text, start_frame, end_frame, color=color)

                    update_duration_auto()

                    # Only load optional fields if present
                    if "num_samples" in meta_info:
                        gui_num_samples_slider.value = meta_info["num_samples"]
                    if "seed" in meta_info:
                        gui_seed.value = meta_info["seed"]
                    if "diffusion_steps" in meta_info:
                        gui_diffusion_steps_slider.value = meta_info["diffusion_steps"]
                    if "cfg" in meta_info:
                        cfg = meta_info["cfg"]
                        if "enabled" in cfg:
                            gui_cfg_checkbox.value = cfg["enabled"]
                        if "text_weight" in cfg:
                            gui_cfg_text_weight_slider.value = cfg["text_weight"]
                        if "constraint_weight" in cfg:
                            gui_cfg_constraint_weight_slider.value = cfg["constraint_weight"]

                # Set frame to 0 when example is loaded.
                session.frame_idx = 0
                event_client.timeline.set_current_frame(0)
                demo.set_frame(event_client.client_id, 0)

                event_client.add_notification(
                    title="Example loaded!",
                    body=f"Loaded example from {example_path}",
                    auto_close_seconds=5.0,
                    color="green",
                )
            except Exception as e:
                import traceback

                traceback.print_exc()
                event_client.add_notification(
                    title="Failed to load example!",
                    body=str(e),
                    auto_close_seconds=10.0,
                    color="red",
                )

        @gui_load_example_button.on_click
        def _(event: viser.GuiEvent) -> None:
            event_client = event.client
            session = get_active_session(event_client)
            if session is None:
                return

            if not session.example_dict or (gui_examples_dropdown.value not in session.example_dict):
                event_client.add_notification(
                    title="No examples available",
                    body="No examples found for the selected model.",
                    auto_close_seconds=5.0,
                    color="red",
                )
                return

            example_path = session.example_dict[gui_examples_dropdown.value]
            load_example_from_path(event_client, example_path, gui_load_gt_checkbox.value)

        @gui_load_example_from_path_button.on_click
        def _(event: viser.GuiEvent) -> None:
            event_client = event.client
            session = get_active_session(event_client)
            if session is None:
                return

            example_path = gui_load_example_path_text.value
            if not example_path:
                event_client.add_notification(
                    title="No example path",
                    body="Please provide an example directory.",
                    auto_close_seconds=5.0,
                    color="red",
                )
                return
            load_example_from_path(event_client, example_path, gui_load_gt_checkbox.value)

        @gui_cfg_checkbox.on_update
        def _(_) -> None:
            if not demo.client_active(client_id):
                return
            val = gui_cfg_checkbox.value
            gui_cfg_text_weight_slider.visible = val
            gui_cfg_constraint_weight_slider.visible = val

        def exit_editing_mode(session: ClientSession):
            gui_edit_constraint_button.label = "Enter Editing Mode"
            gui_generate_button.disabled = False
            gui_generate_button.label = "Generate"
            gui_reset_constraint_button.disabled = True
            if "g1" in session.model_name:
                gui_gizmo_space_dropdown.value = "Local"
                gui_gizmo_space_dropdown.disabled = True
                gui_gizmo_space_dropdown.visible = False
            else:
                gui_gizmo_space_dropdown.disabled = False
                gui_gizmo_space_dropdown.visible = True
            gui_undo_drag_button.disabled = True
            gui_use_soma_layer_checkbox.disabled = False
            session.edit_mode_snapshot = None
            session.undo_drag_snapshot = None

            motion = list(session.motions.values())[0]
            motion.clear_all_gizmos()
            motion.character.set_skinned_mesh_wireframe(False)
            motion.character.set_skeleton_visibility(False)
            motion.character.set_skinned_mesh_visibility(True)
            motion.character.set_skinned_mesh_opacity(1.0)
            session.gui_elements.gui_viz_skinned_mesh_opacity_slider.value = 1.0

            # If the path is dense, put the motion back on the path
            if "2D Root" in session.constraints and session.constraints["2D Root"].dense_path:
                _update_dense_path(motion, session)

            gui_viz_skinned_mesh_checkbox.value = True
            gui_viz_skeleton_checkbox.value = False

        # enter editing mode callback
        @gui_edit_constraint_button.on_click
        def _(event: viser.GuiEvent) -> None:
            event_client = event.client
            session = get_active_session(event_client)
            if session is None:
                return

            session.edit_mode = not session.edit_mode

            edit_alert = "Entered editing mode"
            no_edit_alert = "Exited editing mode"
            edit_message = "You can now modify pose or path constraints."
            no_edit_message = "Can now generate motions."
            event_client.add_notification(
                title=edit_alert if session.edit_mode else no_edit_alert,
                body=edit_message if session.edit_mode else no_edit_message,
                auto_close_seconds=10.0,
                color="blue",
            )

            if session.edit_mode:
                gui_edit_constraint_button.label = "Exit Editing Mode"
                gui_generate_button.disabled = True
                gui_generate_button.label = "Generate Disabled In Editing Mode"
                if "g1" in session.model_name:
                    gui_gizmo_space_dropdown.value = "Local"
                gui_gizmo_space_dropdown.disabled = True
                gui_use_soma_layer_checkbox.disabled = True

                assert len(session.motions) == 1, "Only one motion allowed in edit mode"
                motion = list(session.motions.values())[0]
                snapshot_frame_idx = min(session.frame_idx, motion.length - 1)
                session.edit_mode_snapshot = {}
                ensure_edit_snapshot(session, motion, snapshot_frame_idx)
                gui_reset_constraint_button.disabled = False

                motion.character.set_skeleton_visibility(True)
                # motion.character.set_skinned_mesh_wireframe(True)
                motion.character.set_skinned_mesh_opacity(0.65)
                session.gui_elements.gui_viz_skinned_mesh_opacity_slider.value = 0.65
                motion.character.set_skinned_mesh_visibility(True)
                gui_viz_skinned_mesh_checkbox.value = True
                gui_viz_skeleton_checkbox.value = True

                # need gizmos for root translation and individual joints
                def _on_root2d_gizmo_release():
                    if "2D Root" in session.constraints and session.constraints["2D Root"].dense_path:
                        mot = list(session.motions.values())[0]
                        _update_dense_path(mot, session)

                def _on_gizmo_drag_start():
                    mot = list(session.motions.values())[0]
                    frame_idx = min(session.frame_idx, mot.length - 1)
                    session.undo_drag_snapshot = {
                        "frame_idx": frame_idx,
                        "joints_pos": mot.get_joints_pos(frame_idx),
                        "joints_rot": mot.get_joints_rot(frame_idx),
                    }
                    gui_undo_drag_button.disabled = False

                motion.add_root_translation_gizmo(
                    session.constraints,
                    on_2d_root_drag_end=_on_root2d_gizmo_release,
                    on_drag_start=_on_gizmo_drag_start,
                )
                gizmo_space = "local" if "g1" in session.model_name else gui_gizmo_space_dropdown.value.lower()
                motion.add_joint_gizmos(
                    session.constraints,
                    space=gizmo_space,
                    on_drag_start=_on_gizmo_drag_start,
                )
            else:
                exit_editing_mode(session)

        @gui_reset_constraint_button.on_click
        def _(event: viser.GuiEvent) -> None:
            event_client = event.client
            session = get_active_session(event_client)
            if session is None or not session.edit_mode_snapshot:
                return

            if not session.motions:
                return
            motion = list(session.motions.values())[0]
            snapshot_frame_idx = min(session.frame_idx, motion.length - 1)
            if snapshot_frame_idx not in session.edit_mode_snapshot:
                return
            motion.update_pose_at_frame(
                snapshot_frame_idx,
                joints_pos=session.edit_mode_snapshot[snapshot_frame_idx]["joints_pos"],
                joints_rot=session.edit_mode_snapshot[snapshot_frame_idx]["joints_rot"],
            )
            demo.set_frame(event_client.client_id, snapshot_frame_idx, update_timeline=False)

        @gui_undo_drag_button.on_click
        def _(event: viser.GuiEvent) -> None:
            event_client = event.client
            session = get_active_session(event_client)
            if session is None or session.undo_drag_snapshot is None:
                return

            if not session.motions:
                return
            motion = list(session.motions.values())[0]
            frame_idx = session.undo_drag_snapshot["frame_idx"]
            motion.update_pose_at_frame(
                frame_idx,
                joints_pos=session.undo_drag_snapshot["joints_pos"],
                joints_rot=session.undo_drag_snapshot["joints_rot"],
            )
            demo.set_frame(event_client.client_id, frame_idx, update_timeline=False)
            session.undo_drag_snapshot = None
            gui_undo_drag_button.disabled = True

        def validate_interval(start_frame_idx: int, end_frame_idx: int, max_frame_idx: int) -> bool:
            if start_frame_idx < 0 or start_frame_idx > max_frame_idx:
                return False
            if end_frame_idx < 0 or end_frame_idx > max_frame_idx:
                return False
            if end_frame_idx < start_frame_idx:
                return False
            return True

        def clamp_interval_to_range(
            start_frame_idx: int, end_frame_idx: int, max_frame_idx: int
        ) -> Optional[tuple[int, int]]:
            if end_frame_idx < 0 or start_frame_idx > max_frame_idx:
                return None
            start_clamped = max(0, start_frame_idx)
            end_clamped = min(max_frame_idx, end_frame_idx)
            if end_clamped < start_clamped:
                return None
            return start_clamped, end_clamped

        # add constraint callback
        def add_constraint_callback(
            constraint_id: str,
            constraint_type: str,
            frame_range: tuple[int, int],
            joint_names: list[str] = None,
            verbose: bool = True,
        ):
            """Add a constraint to the session.

            Args:
                constraint_type: str, the type of constraint to add
                frame_range: tuple[int, int], the frame range to add the constraint to
                joint_names: list[str], the names of the joints to constraint if the constraint type is End-Effectors
            """
            # Check if session still exists
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]

            assert len(session.motions) == 1, "Only one motion allowed for adding constraints"
            motion = list(session.motions.values())[0]

            end_effector_type = None
            if constraint_type in [
                "Left Hand",
                "Right Hand",
                "Left Foot",
                "Right Foot",
            ]:
                joint_names = [constraint_type.replace(" ", ""), "Hips"]
                # Hips are required because of smooth root representation
                end_effector_type = constraint_type.replace(" ", "-").lower()
                constraint_type = "End-Effectors"

            # check to make sure interval is valid
            is_interval = frame_range[1] != frame_range[0]
            start_frame_idx = int(frame_range[0])
            end_frame_idx = int(frame_range[1])

            if is_interval:
                clamped = clamp_interval_to_range(start_frame_idx, end_frame_idx, session.max_frame_idx)
                if clamped is None:
                    print("Interval outside range! Couldn't add constraint.")
                    return
                start_frame_idx, end_frame_idx = clamped
            else:
                if not validate_interval(start_frame_idx, end_frame_idx, session.max_frame_idx):
                    print("Invalid interval! Couldn't add constraint.")
                    return

            # collect input args for the constraint based on which track it is
            if is_interval:
                constraint_kwargs = {
                    "interval_id": constraint_id,
                    "start_frame_idx": start_frame_idx,
                    "end_frame_idx": end_frame_idx,
                }
            else:
                constraint_kwargs = {
                    "keyframe_id": constraint_id,
                    "frame_idx": start_frame_idx,
                }

            if constraint_type in ["Full-Body", "End-Effectors"]:
                constraint_kwargs["joints_pos"] = motion.get_joints_pos(start_frame_idx, end_frame_idx)
                constraint_kwargs["joints_rot"] = motion.get_joints_rot(start_frame_idx, end_frame_idx)
                if constraint_type == "End-Effectors":
                    constraint_kwargs["joint_names"] = joint_names
                    constraint_kwargs["end_effector_type"] = end_effector_type

            elif constraint_type == "2D Root":
                constraint_kwargs["root_pos"] = motion.get_projected_root_pos(start_frame_idx, end_frame_idx)

            # add the keyframe(s) to the constraint track
            constraint = session.constraints[constraint_type]
            if is_interval:
                constraint.add_interval(**constraint_kwargs)
            else:
                constraint.add_keyframe(**constraint_kwargs)

            apply_constraint_overlay_visibility(session)

            if verbose:
                client.add_notification(
                    title="Constraint added",
                    body="",
                    auto_close_seconds=5.0,
                    color="blue",
                )

        # timeline callbacks for keyframes and intervals
        @client.timeline.on_keyframe_add
        def _(keyframe_id: str, track_id: str, frame: int):
            """Called when a keyframe is added to a track."""
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]
            with session.timeline_data["keyframe_update_lock"]:
                constraint_type = session.timeline_data["tracks"][track_id]["name"]
                add_constraint_callback(
                    keyframe_id,
                    constraint_type,
                    (frame, frame),
                    verbose=False,
                )
                keyframe_data = client.timeline._keyframes.get(keyframe_id)
                session.timeline_data["keyframes"][keyframe_id] = {
                    "frame": frame,
                    "track_id": track_id,
                    "locked": bool(keyframe_data.locked) if keyframe_data is not None else False,
                    "opacity": keyframe_data.opacity if keyframe_data is not None else 1.0,
                    "value": keyframe_data.value if keyframe_data is not None else None,
                }
                # Update smooth path when adding a keyframe (single action, not drag).
                if constraint_type == "2D Root" and session.constraints["2D Root"].dense_path:
                    motion = list(session.motions.values())[0]
                    _update_dense_path(motion, session)

        @client.timeline.on_interval_add
        def handle_interval_add(interval_id: str, track_id: str, start_frame: int, end_frame: int):
            """Called when an interval is added to a track."""
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]
            with session.timeline_data["keyframe_update_lock"]:
                constraint_type = session.timeline_data["tracks"][track_id]["name"]
                add_constraint_callback(
                    interval_id,
                    constraint_type,
                    (start_frame, end_frame),
                    verbose=False,
                )
                interval_data = client.timeline._intervals.get(interval_id)
                session.timeline_data["intervals"][interval_id] = {
                    "track_id": track_id,
                    "start_frame_idx": start_frame,
                    "end_frame_idx": end_frame,
                    "locked": bool(interval_data.locked) if interval_data is not None else False,
                    "opacity": interval_data.opacity if interval_data is not None else 1.0,
                    "value": interval_data.value if interval_data is not None else None,
                }
                if constraint_type == "2D Root" and session.constraints["2D Root"].dense_path:
                    motion = list(session.motions.values())[0]
                    _update_dense_path(motion, session)

        def remove_constraint_callback(
            constraint_id: str,
            constraint_type: str,
            frame_range: tuple[int, int],
            verbose: bool = True,
        ) -> None:
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]
            session.updating_motions = True

            is_interval = frame_range[1] != frame_range[0]
            start_frame_idx = int(frame_range[0])
            end_frame_idx = int(frame_range[1])

            if is_interval:
                clamped = clamp_interval_to_range(start_frame_idx, end_frame_idx, session.max_frame_idx)
                if clamped is None:
                    return
                start_frame_idx, end_frame_idx = clamped
            else:
                if not validate_interval(start_frame_idx, end_frame_idx, session.max_frame_idx):
                    print("Invalid interval! Couldn't remove constraint.")
                    return

            if constraint_type in [
                "Left Hand",
                "Right Hand",
                "Left Foot",
                "Right Foot",
            ]:
                constraint_type = "End-Effectors"

            constraint = session.constraints[constraint_type]
            if is_interval:
                constraint.remove_interval(constraint_id, start_frame_idx, end_frame_idx)
            else:
                constraint.remove_keyframe(constraint_id, start_frame_idx)

            if verbose:
                client.add_notification(
                    title="Constraint removed",
                    body="",
                    auto_close_seconds=5.0,
                    color="blue",
                )

        @client.timeline.on_keyframe_move
        def handle_keyframe_move(keyframe_id: str, new_frame: int):
            """Called when a keyframe is moved to a new frame."""
            # print(f"Keyframe moved: {keyframe_id} to frame {new_frame}")
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]

            # Cancel any pending timer for this keyframe
            timeline_data = session.timeline_data
            with timeline_data["keyframe_update_lock"]:
                if keyframe_id in timeline_data["keyframe_move_timers"]:
                    timeline_data["keyframe_move_timers"][keyframe_id].cancel()

                # Store the latest target frame
                timeline_data["pending_keyframe_moves"][keyframe_id] = new_frame
                # Create a new timer to execute the actual move after a delay
                # This debounces rapid movements - only execute when user stops moving
                timer = threading.Timer(
                    0.03,  # 10ms delay - adjust as needed
                    _execute_keyframe_move,
                    args=(client_id, keyframe_id, new_frame, session),
                )
                timeline_data["keyframe_move_timers"][keyframe_id] = timer
                timer.start()

        def _execute_keyframe_move(
            client_id: int,
            keyframe_id: str,
            new_frame: int,
            session: ClientSession,
        ):
            """Actually execute the keyframe move operation (called after debounce delay)."""

            timeline_data = session.timeline_data
            with timeline_data["keyframe_update_lock"]:
                # Check if this move is still the latest one
                if keyframe_id not in timeline_data["pending_keyframe_moves"]:
                    return  # Move was cancelled

                if timeline_data["pending_keyframe_moves"][keyframe_id] != new_frame:
                    return  # A newer move superseded this one

                # Remove from pending
                del timeline_data["pending_keyframe_moves"][keyframe_id]
                if keyframe_id in timeline_data["keyframe_move_timers"]:
                    del timeline_data["keyframe_move_timers"][keyframe_id]

                # Now execute the actual move (keep it in the lock so we don't delete it while moving)
                if keyframe_id not in timeline_data["keyframes"]:
                    # double check
                    return
                keyframe_data = timeline_data["keyframes"][keyframe_id]
                if not keyframe_data:
                    return

                # if the frame did not move, don't do anything
                if keyframe_data["frame"] == new_frame:
                    return

                track_id = keyframe_data["track_id"]
                constraint_type = timeline_data["tracks"][track_id]["name"]
                cur_frame = keyframe_data["frame"]

                # Remove constraint at old frame
                remove_constraint_callback(
                    keyframe_id,
                    constraint_type,
                    (cur_frame, cur_frame),
                    verbose=False,
                )
                # Add constraint at new frame
                add_constraint_callback(
                    keyframe_id,
                    constraint_type,
                    (new_frame, new_frame),
                    verbose=False,
                )

                # update our data
                keyframe_data["frame"] = new_frame

                # Schedule path update only after user stops dragging (no move for 300ms).
                if constraint_type == "2D Root":
                    _schedule_dense_path_after_release(session)

        @client.timeline.on_keyframe_delete
        def handle_keyframe_delete(keyframe_id: str):
            """Called when a keyframe is deleted."""
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]
            with session.timeline_data["keyframe_update_lock"]:
                if keyframe_id not in session.timeline_data["keyframes"]:
                    return
                keyframe_data = session.timeline_data["keyframes"][keyframe_id]
                track_id = keyframe_data["track_id"]
                constraint_type = session.timeline_data["tracks"][track_id]["name"]
                cur_frame = keyframe_data["frame"]
                remove_constraint_callback(
                    keyframe_id,
                    constraint_type,
                    (cur_frame, cur_frame),
                    verbose=False,
                )
                del session.timeline_data["keyframes"][keyframe_id]
                if constraint_type == "2D Root" and session.constraints["2D Root"].dense_path:
                    motion = list(session.motions.values())[0]
                    _update_dense_path(motion, session)

        @client.timeline.on_interval_move
        def handle_interval_move(interval_id: str, new_start: int, new_end: int):
            """Called when an interval is moved or resized."""
            # print(f"Interval moved: {interval_id} to {new_start}-{new_end}")
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]

            # Cancel any pending timer for this interval
            # We share the same lock for keyframe and interval moves assuming the user can't move both at the same time
            timeline_data = session.timeline_data
            with timeline_data["keyframe_update_lock"]:
                if interval_id in timeline_data["keyframe_move_timers"]:
                    timeline_data["keyframe_move_timers"][interval_id].cancel()

                # Store the latest target frame
                new_interval = (new_start, new_end)
                timeline_data["pending_keyframe_moves"][interval_id] = new_interval
                # Create a new timer to execute the actual move after a delay
                # This debounces rapid movements - only execute when user stops moving
                timer = threading.Timer(
                    0.5,  # 100ms delay - adding interval is much slower than moving a keyframe
                    _execute_interval_move,
                    args=(client_id, interval_id, new_interval, session),
                )
                timeline_data["keyframe_move_timers"][interval_id] = timer
                timer.start()

        def _execute_interval_move(
            client_id: int,
            interval_id: str,
            new_interval: tuple[int, int],
            session: ClientSession,
        ):
            """Actually execute the interval move operation (called after debounce delay)."""

            timeline_data = session.timeline_data
            with timeline_data["keyframe_update_lock"]:
                # Check if this move is still the latest one
                if interval_id not in timeline_data["pending_keyframe_moves"]:
                    return  # Move was cancelled

                if timeline_data["pending_keyframe_moves"][interval_id] != new_interval:
                    return  # A newer move superseded this one

                # Remove from pending
                del timeline_data["pending_keyframe_moves"][interval_id]
                if interval_id in timeline_data["keyframe_move_timers"]:
                    del timeline_data["keyframe_move_timers"][interval_id]

                # Now execute the actual move
                if interval_id not in timeline_data["intervals"]:
                    return
                interval_data = timeline_data["intervals"][interval_id]
                if not interval_data:
                    return

                # if the interval did not move, don't do anything
                if (
                    interval_data["start_frame_idx"] == new_interval[0]
                    and interval_data["end_frame_idx"] == new_interval[1]
                ):
                    return

                track_id = interval_data["track_id"]
                constraint_type = timeline_data["tracks"][track_id]["name"]
                cur_range = (
                    interval_data["start_frame_idx"],
                    interval_data["end_frame_idx"],
                )

                # Remove constraint at old frame
                remove_constraint_callback(
                    interval_id,
                    constraint_type,
                    cur_range,
                    verbose=False,
                )
                # Add constraint at new frame
                add_constraint_callback(
                    interval_id,
                    constraint_type,
                    new_interval,
                    verbose=False,
                )

                # update our data
                interval_data["start_frame_idx"] = new_interval[0]
                interval_data["end_frame_idx"] = new_interval[1]

                # Schedule path update only after user stops dragging (no move for 300ms).
                if constraint_type == "2D Root":
                    _schedule_dense_path_after_release(session)

        @client.timeline.on_interval_delete
        def handle_interval_delete(interval_id: str):
            """Called when an interval is deleted."""
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]
            with session.timeline_data["keyframe_update_lock"]:
                if interval_id not in session.timeline_data["intervals"]:
                    return
                interval_data = session.timeline_data["intervals"][interval_id]
                track_id = interval_data["track_id"]
                constraint_type = session.timeline_data["tracks"][track_id]["name"]
                remove_constraint_callback(
                    interval_id,
                    constraint_type,
                    (
                        interval_data["start_frame_idx"],
                        interval_data["end_frame_idx"],
                    ),
                    verbose=False,
                )
                del session.timeline_data["intervals"][interval_id]
                if constraint_type == "2D Root" and session.constraints["2D Root"].dense_path:
                    motion = list(session.motions.values())[0]
                    _update_dense_path(motion, session)

        @gui_snap_to_constraint_button.on_click
        def _(event: viser.GuiEvent) -> None:
            event_client = event.client
            session = get_active_session(event_client)
            if session is None:
                return

            target_character_motion = list(session.motions.values())[0]
            frame_idx = session.frame_idx

            if frame_idx >= target_character_motion.length:
                # frame idx larger than the motion, could not snap
                return

            for constraint_name in ["Full-Body", "End-Effectors"]:
                if (
                    constraint_name in session.constraints
                    and frame_idx in session.constraints[constraint_name].keyframes
                ):
                    pos = session.constraints[constraint_name].keyframes[frame_idx]["joints_pos"]
                    rot = session.constraints[constraint_name].keyframes[frame_idx]["joints_rot"]

                    # update the full joints_pos of the character to match the constraints
                    target_character_motion.update_pose_at_frame(
                        frame_idx,
                        joints_pos=pos,
                        joints_rot=rot,
                    )
                    target_character_motion.set_frame(frame_idx)
                    return  # motion already fully changed

            if "2D Root" in session.constraints and frame_idx in session.constraints["2D Root"].keyframes:
                # update only the root position
                new_root_pos = session.constraints["2D Root"].keyframes[frame_idx]
                old_root_pos = target_character_motion.get_projected_root_pos(frame_idx)
                root_diff = new_root_pos - old_root_pos
                root_diff[1] = 0.0  # don't change height

                new_joints_pos = (
                    target_character_motion.joints_pos[frame_idx]
                    + to_torch(
                        root_diff,
                        device=target_character_motion.joints_pos.device,
                        dtype=target_character_motion.joints_pos.dtype,
                    )[None]
                )
                rot = target_character_motion.joints_rot[frame_idx]

                target_character_motion.update_pose_at_frame(
                    frame_idx,
                    joints_pos=new_joints_pos,
                    joints_rot=rot,
                )
                target_character_motion.set_frame(frame_idx)

        @gui_clear_all_constraints_button.on_click
        def _(event: viser.GuiEvent) -> None:
            event_client = event.client
            session = get_active_session(event_client)
            if session is None:
                return
            with session.timeline_data["keyframe_update_lock"]:
                # use the lock here to wait for any constraint updates to finish
                for constraint in list(session.constraints.values()):
                    constraint.clear()
                client.timeline.clear_keyframes()
                client.timeline.clear_intervals()
            if gui_dense_path_checkbox.value:
                gui_dense_path_checkbox.value = False
                if "2D Root" in session.constraints:
                    session.constraints["2D Root"].set_dense_path(False)

        # generation callback
        @gui_generate_button.on_click
        def _(event: viser.GuiEvent) -> None:
            event_client = event.client
            session = get_active_session(event_client)
            if session is None:
                return

            generating_notif = event_client.add_notification(
                title="Generating motion...",
                body="Generating motions for the given prompt!",
                loading=True,
                with_close_button=False,
            )
            gui_generate_button.disabled = True
            client.timeline.disable_constraints()

            num_samples = gui_num_samples_slider.value
            timeline = session.client.timeline

            # sort them to avoid issues:
            prompt_values = sorted([x for x in timeline._prompts.values()], key=lambda x: x.start_frame)

            texts = [x.text for x in prompt_values]
            num_frames = compute_prompt_num_frames(prompt_values)

            # compute the total duration
            total_nb_frames = sum(num_frames)
            total_duration = total_nb_frames / session.model_fps

            # update just in case
            set_new_duration(client_id, total_duration)

            transitions_parameters = {
                "num_transition_frames": gui_num_transition_frames_slider.value,
                "share_transition": gui_share_transition_checkbox.value,
                "percentage_transition_override": gui_percentage_transition_sharing_slider.value / 100,
            }

            # G1: postprocessing is disabled (does not work well for this model).
            postprocess_parameters = {
                "post_processing": (False if "g1" in session.model_name else gui_postprocess_checkbox.value),
                "root_margin": gui_root_margin.value,
            }
            try:
                demo.generate(
                    event_client,
                    texts,
                    num_frames,
                    num_samples,
                    gui_seed.value,
                    gui_diffusion_steps_slider.value,
                    cfg_weight=[
                        gui_cfg_text_weight_slider.value,
                        gui_cfg_constraint_weight_slider.value,
                    ],
                    cfg_type="separated" if gui_cfg_checkbox.value else "nocfg",
                    postprocess_parameters=postprocess_parameters,
                    transitions_parameters=transitions_parameters,
                    real_robot_rotations=gui_real_robot_rotations_checkbox.value,
                )
                session.max_frame_idx = int(session.cur_duration * session.model_fps - 1)
                session.max_frame_idx = int(session.cur_duration * session.model_fps) - 1
                if session.frame_idx > session.max_frame_idx:
                    session.frame_idx = session.max_frame_idx

                if num_samples > 1:
                    # add mesh selector to choose character to commit
                    def commit_motion(event: viser.GuiEvent) -> None:
                        target = event.target
                        commit_name = target.name.split("/")[1]  # e.g. /character0/simple_skinned
                        print(f"Committing motion for character: {commit_name}")
                        # delete non-selected motions
                        new_motion_kwargs = None
                        for character_name, motion in session.motions.items():
                            if character_name == commit_name:
                                new_motion_kwargs = {
                                    "skeleton": session.skeleton,
                                    "joints_rot": motion.joints_rot,
                                    "foot_contacts": motion.foot_contacts,
                                }
                                root_x_offset = motion.joints_pos[0, session.skeleton.root_idx, 0]
                                new_joints_pos = motion.joints_pos.clone()
                                new_joints_pos[..., 0] -= root_x_offset
                                new_motion_kwargs["joints_pos"] = new_joints_pos
                                break
                        # clear and re-add the selected motion
                        demo.clear_motions(event_client.client_id)
                        demo.add_character_motion(event_client, **new_motion_kwargs)
                        gui_edit_constraint_button.disabled = False
                        gui_generate_button.disabled = False
                        gui_snap_to_constraint_button.disabled = False
                        client.timeline.enable_constraints()
                        gui_generate_button.label = "Generate"
                        gui_save_example_button.disabled = False
                        gui_save_motion_button.disabled = False
                        gui_download_button.disabled = False
                        gui_save_constraints_button.disabled = False
                        gui_load_example_button.disabled = False

                    for motion in session.motions.values():
                        char = motion.character
                        character_name = char.name  # e.g. "character0"
                        if char.skinned_mesh is not None:
                            char.skinned_mesh.on_click(commit_motion)
                        elif char.g1_mesh_rig is not None:
                            # Register click on every part so any part can be clicked,
                            # and use highlight_group so the whole robot highlights together.
                            for handle in char.g1_mesh_rig.mesh_handles:
                                handle.on_click(commit_motion, highlight_group=character_name)

                    gui_edit_constraint_button.disabled = True
                    gui_generate_button.disabled = True
                    gui_snap_to_constraint_button.disabled = True
                    gui_generate_button.label = "Choose Sample Before Generating"
                    gui_save_example_button.disabled = True
                    gui_save_motion_button.disabled = True
                    gui_download_button.disabled = True
                    gui_save_constraints_button.disabled = True
                    gui_load_example_button.disabled = True
                else:
                    gui_edit_constraint_button.disabled = False
                    gui_generate_button.disabled = False
                    gui_snap_to_constraint_button.disabled = False
                    client.timeline.enable_constraints()

                generating_notif.title = "Motion generation finished!"
                generating_notif.body = "Motions have been generated successfully for the given prompt."
                if num_samples > 1:
                    generating_notif.body += " Now choose which sample to commit."
                generating_notif.loading = False
                generating_notif.with_close_button = True
                generating_notif.auto_close_seconds = 5.0
                generating_notif.color = "green"

                # put the motion at zero
                demo.set_frame(client_id, 0)

            except Exception as e:
                raise e
                # If client disconnected or any other error occurred
                if event_client.client_id in demo.client_sessions:
                    session = demo.client_sessions[event_client.client_id]
                    gui_generate_button.disabled = False
                    gui_load_example_button.disabled = False
                    gui_save_example_button.disabled = False
                    gui_save_motion_button.disabled = False
                    gui_download_button.disabled = False
                    event_client.add_notification(
                        title="Generation failed!",
                        body=f"Error: {str(e)}",
                        auto_close_seconds=5.0,
                        color="red",
                    )
                print(f"Error during generation for client {event_client.client_id}: {e}")

    #
    # Visualization settings
    #
    with tab_group.add_tab("Visualize", viser.Icon.EYE):
        with client.gui.add_folder("Playback", expand_by_default=True):
            gui_model_fps = client.gui.add_number("Model FPS", initial_value=model_fps, disabled=True)
            gui_playback_speed_buttons = client.gui.add_button_group(
                "Playback Speed",
                options=[
                    "0.5x",
                    "1x",
                    "2x",
                ],
            )
            gui_playback_speed_buttons.value = "1x"

            @client.timeline.on_frame_change
            def handle_timeline_frame_change(new_frame_idx: int):
                """Update the frame when the user clicks on the timeline."""
                demo.set_frame(client_id, new_frame_idx, update_timeline=False)
                session = demo.client_sessions.get(client_id)
                if session is not None:
                    if session.edit_mode and session.motions:
                        motion = list(session.motions.values())[0]
                        snapshot_frame_idx = min(session.frame_idx, motion.length - 1)
                        ensure_edit_snapshot(session, motion, snapshot_frame_idx)
                    update_snap_to_constraint_button(session)

            @client.timeline.on_prompt_add
            async def _on_add(
                prompt_id: str,
                start_frame: int,
                end_frame: int,
                text: str,
                color: tuple[int, int, int] | None,
            ) -> None:
                update_duration_auto()

            @client.timeline.on_prompt_update
            async def _on_update(prompt_id: str, new_text: str) -> None:
                update_duration_auto()

            @client.timeline.on_prompt_resize
            async def _on_resize(prompt_id: str, new_start: int, new_end: int) -> None:
                update_duration_auto()

            @client.timeline.on_prompt_move
            async def _on_move(prompt_id: str, new_start: int, new_end: int) -> None:
                update_duration_auto()

            @client.timeline.on_prompt_delete
            async def _on_delete(prompt_id: str) -> None:
                update_duration_auto()

            def play_pause_button_callback(session: ClientSession):
                session.playing = not session.playing

            def next_frame_callback(session: ClientSession):
                if session.frame_idx < session.max_frame_idx:
                    session.frame_idx += 1
                if session.frame_idx == session.max_frame_idx:
                    pass
                demo.set_frame(client_id, session.frame_idx)

            def prev_frame_callback(session: ClientSession):
                if session.frame_idx > 0:
                    session.frame_idx -= 1
                if session.frame_idx == 0:
                    pass
                demo.set_frame(client_id, session.frame_idx)

            @gui_playback_speed_buttons.on_click
            def _(_) -> None:
                if not demo.client_active(client_id):
                    return
                speed_map = {
                    "0.5x": 0.5,
                    "1x": 1.0,
                    "2x": 2.0,
                }
                session = demo.client_sessions[client_id]
                session.playback_speed = speed_map[gui_playback_speed_buttons.value]

        with client.gui.add_folder("Body options", expand_by_default=True):
            gui_viz_skinned_mesh_checkbox = client.gui.add_checkbox("Show Mesh", initial_value=True)
            gui_viz_skinned_mesh_opacity_slider = client.gui.add_slider(
                "Mesh Opacity", min=0.0, max=1.0, step=0.01, initial_value=1.0
            )
            gui_viz_skeleton_checkbox = client.gui.add_checkbox("Show Skeleton", initial_value=False)
            gui_viz_foot_contacts_checkbox = client.gui.add_checkbox("Show Foot Contacts", initial_value=False)
            gui_viz_foot_contacts_checkbox.visible = gui_viz_skeleton_checkbox.value
        with client.gui.add_folder("Camera options", expand_by_default=True):
            gui_camera_fov_slider = client.gui.add_slider(
                "Camera FOV (deg)",
                min=30.0,
                max=90.0,
                step=1.0,
                initial_value=45.0,
            )
            client.camera.fov = np.deg2rad(gui_camera_fov_slider.value)
        with client.gui.add_folder("Interface options", expand_by_default=True):
            gui_show_timeline_checkbox = client.gui.add_checkbox(
                "Show Timeline",
                initial_value=True,
            )
            gui_show_constraint_tracks_checkbox = client.gui.add_checkbox(
                "Show Constraint tracks",
                initial_value=True,
            )
            gui_show_constraint_labels_checkbox = client.gui.add_checkbox(
                "Show Constraint labels",
                initial_value=True,
            )
            gui_show_starting_direction_checkbox = client.gui.add_checkbox(
                "Show Starting Direction",
                initial_value=True,
            )
            gui_dark_mode_checkbox = client.gui.add_checkbox(
                "Dark Mode",
                initial_value=False,  # Default to light mode
            )
            gui_show_constraint_tracks_checkbox.visible = gui_show_timeline_checkbox.value
            demo.set_start_direction_visible(client_id, gui_show_starting_direction_checkbox.value)

        @gui_dark_mode_checkbox.on_update
        def _(_):
            # Apply the theme using configure_theme (pass uuid so titlebar toggle stays)
            demo.configure_theme(
                client,
                gui_dark_mode_checkbox.value,
                titlebar_dark_mode_checkbox_uuid=gui_dark_mode_checkbox.uuid,
            )
            session = demo.client_sessions[client.client_id]
            for motion in session.motions.values():
                motion.character.change_theme(gui_dark_mode_checkbox.value)

        # Show dark mode toggle in titlebar (right of Github), hide sidebar checkbox
        demo.configure_theme(
            client,
            gui_dark_mode_checkbox.value,
            titlebar_dark_mode_checkbox_uuid=gui_dark_mode_checkbox.uuid,
        )
        gui_dark_mode_checkbox.visible = False

        @gui_show_constraint_labels_checkbox.on_update
        def _(_):
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]
            for constraint in session.constraints.values():
                constraint.set_label_visibility(gui_show_constraint_labels_checkbox.value)

        @gui_show_timeline_checkbox.on_update
        def _(_):
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]
            session.client.timeline.set_visible(gui_show_timeline_checkbox.value)
            gui_show_constraint_tracks_checkbox.visible = gui_show_timeline_checkbox.value
            if gui_show_timeline_checkbox.value:
                demo.set_constraint_tracks_visible(session, gui_show_constraint_tracks_checkbox.value)

        @gui_show_constraint_tracks_checkbox.on_update
        def _(_):
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]
            demo.set_constraint_tracks_visible(session, gui_show_constraint_tracks_checkbox.value)

        @gui_show_starting_direction_checkbox.on_update
        def _(_):
            if not demo.client_active(client_id):
                return
            demo.set_start_direction_visible(client_id, gui_show_starting_direction_checkbox.value)

        @gui_viz_skeleton_checkbox.on_update
        def _(_) -> None:
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]
            gui_viz_foot_contacts_checkbox.visible = gui_viz_skeleton_checkbox.value
            if not gui_viz_skeleton_checkbox.value:
                gui_viz_foot_contacts_checkbox.value = False
            for motion in session.motions.values():
                motion.character.set_skeleton_visibility(gui_viz_skeleton_checkbox.value)

        @gui_viz_foot_contacts_checkbox.on_update
        def _(_) -> None:
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]
            for motion in session.motions.values():
                motion.character.set_show_foot_contacts(gui_viz_foot_contacts_checkbox.value)

        @gui_viz_skinned_mesh_checkbox.on_update
        def _(_) -> None:
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]
            for motion in session.motions.values():
                motion.character.set_skinned_mesh_visibility(gui_viz_skinned_mesh_checkbox.value)

        @gui_viz_skinned_mesh_opacity_slider.on_update
        def _(_) -> None:
            if not demo.client_active(client_id):
                return
            session = demo.client_sessions[client_id]
            for motion in session.motions.values():
                motion.character.set_skinned_mesh_opacity(gui_viz_skinned_mesh_opacity_slider.value)

        @gui_camera_fov_slider.on_update
        def _(_) -> None:
            if not demo.client_active(client_id):
                return
            client.camera.fov = np.deg2rad(gui_camera_fov_slider.value)

            #

    # Instructions tab
    #
    with tab_group.add_tab("Instructions", viser.Icon.INFO_CIRCLE):
        client.gui.add_markdown(DEMO_UI_INSTRUCTIONS_TAB_MD)

    #
    # Keyboard events
    #
    space_pressed = [False]

    @client.scene.on_keyboard_event("keydown", debounce_ms=100)
    def handle_key(event: viser.KeyboardEvent) -> None:
        # Check if client session still exists
        if client_id not in demo.client_sessions:
            return

        session = demo.client_sessions[client_id]

        if event.event_type == "keyup":
            if event.key == " ":
                space_pressed[0] = False
            return

        # Space bar: only toggle on FIRST press
        if event.key == " ":
            if not space_pressed[0]:
                space_pressed[0] = True
                play_pause_button_callback(session)
            return

        # Handle arrow keys: frame navigation (fast OS repeat with 50ms debounce).
        elif event.key == "ArrowLeft":
            prev_frame_callback(session)
        elif event.key == "ArrowRight":
            next_frame_callback(session)

    _update_visibility_for_loaded_model(model_name)

    gui_elements = GuiElements(
        gui_play_pause_button=gui_play_pause_button,
        gui_next_frame_button=gui_next_frame_button,
        gui_prev_frame_button=gui_prev_frame_button,
        gui_generate_button=gui_generate_button,
        gui_model_fps=gui_model_fps,
        gui_timeline=gui_timeline,
        gui_viz_skeleton_checkbox=gui_viz_skeleton_checkbox,
        gui_viz_foot_contacts_checkbox=gui_viz_foot_contacts_checkbox,
        gui_viz_skinned_mesh_checkbox=gui_viz_skinned_mesh_checkbox,
        gui_viz_skinned_mesh_opacity_slider=gui_viz_skinned_mesh_opacity_slider,
        gui_camera_fov_slider=gui_camera_fov_slider,
        gui_duration_slider=gui_duration_slider,
        gui_num_samples_slider=gui_num_samples_slider,
        gui_cfg_checkbox=gui_cfg_checkbox,
        gui_cfg_text_weight_slider=gui_cfg_text_weight_slider,
        gui_cfg_constraint_weight_slider=gui_cfg_constraint_weight_slider,
        gui_diffusion_steps_slider=gui_diffusion_steps_slider,
        gui_seed=gui_seed,
        gui_postprocess_checkbox=gui_postprocess_checkbox,
        gui_root_margin=gui_root_margin,
        gui_real_robot_rotations_checkbox=gui_real_robot_rotations_checkbox,
        gui_dark_mode_checkbox=gui_dark_mode_checkbox,
        gui_use_soma_layer_checkbox=gui_use_soma_layer_checkbox,
    )
    return (
        gui_elements,
        timeline_tracks,
        example_dict,
        gui_examples_dropdown,
        gui_save_example_path_text,
        gui_model_selector,
    )
