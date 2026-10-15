/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Compiler specific defines

// Finds the compiler type and version.
#if defined(__clang__)
#    define COMPILER_CLANG
#elif defined(__GNUC__) // Check after Clang, as Clang defines this too
#    define COMPILER_GNUC
#elif defined(_MSC_VER) // Check after Clang, since we could be building with either within VS
#    define COMPILER_MSVC
#else
#    pragma error "Unknown compiler. "
#endif

#if defined(COMPILER_MSVC)
	#define FORCE_INLINE __forceinline
#elif defined(COMPILER_GNUC) || defined(__clang__)
	#define FORCE_INLINE inline __attribute__((always_inline))
#endif
