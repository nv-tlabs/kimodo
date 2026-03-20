/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>
#if defined(__aarch64__) || defined(__ARM_NEON)
	#define SIMDE_ENABLE_NATIVE_ALIASES
    #include <simde/x86/mmx.h>
    #include <simde/x86/sse.h>
    #include <simde/x86/sse2.h>
    #include <simde/x86/sse3.h>
    #include <simde/x86/ssse3.h>
    #include <simde/x86/sse4.1.h>
    #include <simde/x86/avx.h>
#else
	#include <immintrin.h>
#endif

namespace SIMD
{
    struct alignas( 16 ) IntMask
    {
        inline operator __m128( ) const { return reinterpret_cast<__m128 const&>( *this ); }
        inline operator __m128i( ) const { return _mm_castps_si128( *this ); }
        inline operator __m128d( ) const { return _mm_castps_pd( *this ); }

        int32_t i[4];
    };

    struct alignas( 16 ) UIntMask
    {
        inline operator __m128( ) const { return reinterpret_cast<__m128 const&>( *this ); }
        inline operator __m128i( ) const { return _mm_castps_si128( *this ); }
        inline operator __m128d( ) const { return _mm_castps_pd( *this ); }

        uint32_t v[4];
    };

    struct alignas( 16 ) FloatMask
    {
        inline operator __m128() const { return reinterpret_cast<__m128 const&>( *this ); }
        inline operator __m128i() const { return _mm_castps_si128( *this ); }
        inline operator __m128d() const { return _mm_castps_pd( *this ); }

        float v[4];
    };

    // Int Operations
    //-------------------------------------------------------------------------

    namespace Int
    {
        FORCE_INLINE bool Equal( __m128 V1, __m128 V2 )
        {
            __m128i vTemp = _mm_cmpeq_epi32( _mm_castps_si128( V1 ), _mm_castps_si128( V2 ) );
            return ( ( ( _mm_movemask_ps( _mm_castsi128_ps( vTemp ) ) & 7 ) == 7 ) != 0 );
        }

        FORCE_INLINE bool NotEqual( __m128 V1, __m128 V2 )
        {
            __m128i vTemp = _mm_cmpeq_epi32( _mm_castps_si128( V1 ), _mm_castps_si128( V2 ) );
            return ( ( _mm_movemask_ps( _mm_castsi128_ps( vTemp ) ) != 0xF ) != 0 );
        }

        FORCE_INLINE __m128 And( __m128 V1, __m128 V2 )
        {
            return _mm_and_ps( V1, V2 );
        }

        FORCE_INLINE __m128 Or( __m128 V1, __m128 V2 )
        {
            __m128i V = _mm_or_si128( _mm_castps_si128( V1 ), _mm_castps_si128( V2 ) );
            return _mm_castsi128_ps( V );
        }
    }

    //-------------------------------------------------------------------------

    static __m128 const g_sinCoefficients0 = { -0.16666667f, +0.0083333310f, -0.00019840874f, +2.7525562e-06f };
    static __m128 const g_sinCoefficients1 = { -2.3889859e-08f, -0.16665852f, +0.0083139502f, -0.00018524670f };
    static __m128 const g_cosCoefficients0 = { -0.5f, +0.041666638f, -0.0013888378f, +2.4760495e-05f };
    static __m128 const g_cosCoefficients1 = { -2.6051615e-07f, -0.49992746f, +0.041493919f, -0.0012712436f };
    static __m128 const g_tanCoefficients0 = { 1.0f, 0.333333333f, 0.133333333f, 5.396825397e-2f };
    static __m128 const g_tanCoefficients1 = { 2.186948854e-2f, 8.863235530e-3f, 3.592128167e-3f, 1.455834485e-3f };
    static __m128 const g_tanCoefficients2 = { 5.900274264e-4f, 2.391290764e-4f, 9.691537707e-5f, 3.927832950e-5f };
    static __m128 const g_arcCoefficients0 = { +1.5707963050f, -0.2145988016f, +0.0889789874f, -0.0501743046f };
    static __m128 const g_arcCoefficients1 = { +0.0308918810f, -0.0170881256f, +0.0066700901f, -0.0012624911f };
    static __m128 const g_aTanCoefficients0 = { -0.3333314528f, +0.1999355085f, -0.1420889944f, +0.1065626393f };
    static __m128 const g_aTanCoefficients1 = { -0.0752896400f, +0.0429096138f, -0.0161657367f, +0.0028662257f };
    static __m128 const g_aTanEstCoefficients0 = { +0.999866f, +0.999866f, +0.999866f, +0.999866f };
    static __m128 const g_aTanEstCoefficients1 = { -0.3302995f, +0.180141f, -0.085133f, +0.0208351f };
    static __m128 const g_tanEstCoefficients = { 2.484f, -1.954923183e-1f, 2.467401101f, Math::OneDivPi };
    static __m128 const g_arcEstCoefficients = { +1.5707288f,-0.2121144f,+0.0742610f,-0.0187293f };
    static __m128 const g_aTan2Constants = { Math::Pi, Math::PiDivTwo, Math::PiDivFour, 2.3561944905f /* 3/4 Pi */ };

    //-------------------------------------------------------------------------

    static FloatMask const g_noFraction = { 8388608.0f,8388608.0f,8388608.0f,8388608.0f };
    static IntMask const g_absMask = { 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF };
    static UIntMask const g_trueMask = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF };
    static UIntMask const g_signMask = { 0x80000000, 0x80000000, 0x80000000, 0x80000000 };
    static UIntMask const g_maskX000 = { 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000000 };
    static UIntMask const g_mask0Y00 = { 0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000000 };
    static UIntMask const g_mask00Z0 = { 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000 };
    static UIntMask const g_mask000W = { 0x00000000, 0x00000000, 0x00000000, 0xFFFFFFFF };
    static UIntMask const g_maskXY00 = { 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000 };
    static UIntMask const g_maskXYZ0 = { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000 };
}
