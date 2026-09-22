/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "Compiler.h"
#include "Debug.h"

#include "Constants.h"

#include <math.h>
#include <cstdlib>
#include <stdint.h>

//
// Scalar related methods
//

namespace Math
{
    FORCE_INLINE float Sin( float value ) { return sinf( value ); }
    FORCE_INLINE float Cos( float value ) { return cosf( value ); }
    FORCE_INLINE float Tan( float value ) { return tanf( value ); }

    FORCE_INLINE float ASin( float value ) { return asinf( value ); }
    FORCE_INLINE float ACos( float value ) { return acosf( value ); }
    FORCE_INLINE float ATan( float value ) { return atanf( value ); }
    FORCE_INLINE float ATan2( float y, float x ) { return atan2f( y, x ); }

    FORCE_INLINE float Cosec( float value ) { return 1.0f / sinf( value ); }
    FORCE_INLINE float Sec( float value ) { return 1.0f / cosf( value ); }
    FORCE_INLINE float Cot( float value ) { return 1.0f / tanf( PiDivTwo - value ); }

    FORCE_INLINE float Pow( float x, float y ) { return powf( x, y ); }
    FORCE_INLINE float Sqr( float value ) { return value * value; }
    FORCE_INLINE float Sqrt( float value ) { return sqrtf( value ); }

    FORCE_INLINE float Log( float value ) { return logf( value ); }
    FORCE_INLINE float Log2f( float value ) { return log2f( value ); }

    FORCE_INLINE float AddToMovingAverage( float currentAverage, uint64_t numCurrentSamples, float newValue )
    {
        return currentAverage + ( ( newValue - currentAverage ) / float( numCurrentSamples + 1 ) );
    }

    FORCE_INLINE float Abs( float a ) { return fabsf( a ); }
    FORCE_INLINE double Abs( double a ) { return fabs( a ); }
    FORCE_INLINE int8_t Abs( int8_t a ) { return (int8_t) abs( a ); }
    FORCE_INLINE int16_t Abs( int16_t a ) { return (int16_t) abs( a ); }
    FORCE_INLINE int32_t Abs( int32_t a ) { return std::labs( a ); }
    FORCE_INLINE int64_t Abs( int64_t a ) { return std::llabs( a ); }

    FORCE_INLINE float Reciprocal( float r ) { return 1.0f / r; }
    FORCE_INLINE double Reciprocal( double r ) { return 1.0 / r; }

    template<typename T>
    FORCE_INLINE T Min( T a, T b ) { return a <= b ? a : b; }

    template<typename T>
    FORCE_INLINE T Max( T a, T b ) { return a >= b ? a : b; }

    template<typename T>
    FORCE_INLINE T AbsMin( T a, T b ) { return Abs( a ) <= Abs( b ) ? a : b; }

    template<typename T>
    FORCE_INLINE T AbsMax( T a, T b ) { return Abs( a ) >= Abs( b ) ? a : b; }

    template<typename T>
    FORCE_INLINE T Sqrt( T a ) { return sqrt( a ); }

    template<typename T>
    FORCE_INLINE T Clamp( T value, T lowerBound, T upperBound )
    {
        ASSERT( lowerBound <= upperBound );
        return Min( Max( value, lowerBound ), upperBound );
    }

    template<typename T>
    FORCE_INLINE bool IsInRangeInclusive( T value, T lowerBound, T upperBound )
    {
        ASSERT( lowerBound < upperBound );
        return value >= lowerBound && value <= upperBound;
    }

    template<typename T>
    FORCE_INLINE bool IsInRangeExclusive( T value, T lowerBound, T upperBound )
    {
        ASSERT( lowerBound < upperBound );
        return value > lowerBound && value < upperBound;
    }

    // Decomposes a float into integer and remainder portions, remainder is return and the integer result is stored in the integer portion
    FORCE_INLINE float ModF( float value, float& integerPortion )
    {
        return modff( value, &integerPortion );
    }

    // Returns the floating point remainder of x/y
    FORCE_INLINE float FModF( float x, float y )
    {
        return fmodf( x, y );
    }

    template<typename T>
    FORCE_INLINE T Lerp( T A, T B, float t )
    {
        return A + ( B - A ) * t;
    }

    FORCE_INLINE float PercentageThroughRange( float value, float lowerBound, float upperBound )
    {
        ASSERT( lowerBound < upperBound );
        return Clamp( value, lowerBound, upperBound ) / ( upperBound - lowerBound );
    }

    FORCE_INLINE bool IsNearEqual( float value, float comparand, float epsilon = Epsilon )
    {
        return fabsf( value - comparand ) <= epsilon;
    }

    FORCE_INLINE bool IsNearZero( float value, float epsilon = Epsilon )
    {
        return fabsf( value ) <= epsilon;
    }

    FORCE_INLINE bool IsNearEqual( double value, double comparand, double epsilon = Epsilon )
    {
        return fabs( value - comparand ) <= epsilon;
    }

    FORCE_INLINE bool IsNearZero( double value, double epsilon = Epsilon )
    {
        return fabs( value ) <= epsilon;
    }

    FORCE_INLINE float Ceiling( float value )
    {
        return ceilf( value );
    }

    FORCE_INLINE int32_t CeilingToInt( float value )
    {
        return (int32_t) ceilf( value );
    }

    FORCE_INLINE float Floor( float value )
    {
        return floorf( value );
    }

    FORCE_INLINE int32_t FloorToInt( float value )
    {
        return (int32_t) floorf( value );
    }

    FORCE_INLINE float Round( float value )
    {
        return roundf( value );
    }

    FORCE_INLINE int32_t RoundToInt( float value )
    {
        return (int32_t) roundf( value );
    }

    inline float RemapRange( float value, float fromRangeBegin, float fromRangeEnd, float toRangeBegin, float toRangeEnd )
    {
        float const fromRangeLength = fromRangeEnd - fromRangeBegin;
        float const percentageThroughFromRange = Clamp( ( value - fromRangeBegin ) / fromRangeLength, 0.0f, 1.0f );
        float const toRangeLength = toRangeEnd - toRangeBegin;
        float const result = toRangeBegin + ( percentageThroughFromRange * toRangeLength );

        return result;
    }

    FORCE_INLINE float Square( float value )
    {
        return value * value;
    }

    FORCE_INLINE float SmoothStep01( float value )
    {
        value = Clamp( value, 0.0f, 1.0f );
        return value * value * ( 3.0f - 2.0f * value );
    }
}
