{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int8X8 (Int8X8) where

-- This code was AUTOMATICALLY generated, DO NOT EDIT!

import Data.Primitive.SIMD.Class

import GHC.Int

import GHC.Types
import GHC.Exts
import GHC.ST

import Foreign.Storable

import Control.Monad.Primitive

import Data.Primitive.Types
import Data.Primitive.ByteArray
import Data.Primitive.Addr
import Data.Monoid
import Data.Typeable

import qualified Data.Vector.Primitive as PV
import qualified Data.Vector.Primitive.Mutable as PMV
import Data.Vector.Unboxed (Unbox)
import qualified Data.Vector.Unboxed as UV
import Data.Vector.Generic (Vector(..))
import Data.Vector.Generic.Mutable (MVector(..))

-- ** Int8X8
data Int8X8 = Int8X8 Int# Int# Int# Int# Int# Int# Int# Int# deriving Typeable

broadcastInt8# :: Int# -> Int#
broadcastInt8# v = v

packInt8# :: (# Int# #) -> Int#
packInt8# (# v #) = v

unpackInt8# :: Int# -> (# Int# #)
unpackInt8# v = (# v #)

insertInt8# :: Int# -> Int# -> Int# -> Int#
insertInt8# _ v _ = v

negateInt8# :: Int# -> Int#
negateInt8# a = case negate (I8# a) of I8# b -> b

plusInt8# :: Int# -> Int# -> Int#
plusInt8# a b = case I8# a + I8# b of I8# c -> c

minusInt8# :: Int# -> Int# -> Int#
minusInt8# a b = case I8# a - I8# b of I8# c -> c

timesInt8# :: Int# -> Int# -> Int#
timesInt8# a b = case I8# a * I8# b of I8# c -> c

quotInt8# :: Int# -> Int# -> Int#
quotInt8# a b = case I8# a `quot` I8# b of I8# c -> c

remInt8# :: Int# -> Int# -> Int#
remInt8# a b = case I8# a `rem` I8# b of I8# c -> c

abs' :: Int8 -> Int8
abs' (I8# x) = I8# (abs# x)

{-# NOINLINE abs# #-}
abs# :: Int# -> Int#
abs# x = case abs (I8# x) of
    I8# y -> y

signum' :: Int8 -> Int8
signum' (I8# x) = I8# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Int# -> Int#
signum# x = case signum (I8# x) of
    I8# y -> y

instance Eq Int8X8 where
    a == b = case unpackInt8X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackInt8X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8

instance Ord Int8X8 where
    a `compare` b = case unpackInt8X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackInt8X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8

instance Show Int8X8 where
    showsPrec _ a s = case unpackInt8X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> "Int8X8 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (")" ++ s))))))))

instance Num Int8X8 where
    (+) = plusInt8X8
    (-) = minusInt8X8
    (*) = timesInt8X8
    negate = negateInt8X8
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int8X8 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int8X8 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int8X8 where
    type Elem Int8X8 = Int8
    type ElemTuple Int8X8 = (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8)
    nullVector         = broadcastVector 0
    vectorSize  _      = 8
    elementSize _      = 1
    broadcastVector    = broadcastInt8X8
    unsafeInsertVector = unsafeInsertInt8X8
    packVector         = packInt8X8
    unpackVector       = unpackInt8X8
    mapVector          = mapInt8X8
    zipVector          = zipInt8X8
    foldVector         = foldInt8X8

instance SIMDIntVector Int8X8 where
    quotVector = quotInt8X8
    remVector  = remInt8X8

instance Prim Int8X8 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt8X8Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt8X8Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt8X8Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt8X8OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt8X8OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt8X8OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int8X8 = V_Int8X8 (PV.Vector Int8X8)
newtype instance UV.MVector s Int8X8 = MV_Int8X8 (PMV.MVector s Int8X8)

instance Vector UV.Vector Int8X8 where
    basicUnsafeFreeze (MV_Int8X8 v) = V_Int8X8 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int8X8 v) = MV_Int8X8 <$> PV.unsafeThaw v
    basicLength (V_Int8X8 v) = PV.length v
    basicUnsafeSlice start len (V_Int8X8 v) = V_Int8X8(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int8X8 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int8X8 m) (V_Int8X8 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int8X8 where
    basicLength (MV_Int8X8 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int8X8 v) = MV_Int8X8(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int8X8 v) (MV_Int8X8 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int8X8 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int8X8 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int8X8 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int8X8 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int8X8

{-# INLINE broadcastInt8X8 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt8X8 :: Int8 -> Int8X8
broadcastInt8X8 (I8# x) = case broadcastInt8# x of
    v -> Int8X8 v v v v v v v v

{-# INLINE packInt8X8 #-}
-- | Pack the elements of a tuple into a vector.
packInt8X8 :: (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8) -> Int8X8
packInt8X8 (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8) = Int8X8 (packInt8# (# x1 #)) (packInt8# (# x2 #)) (packInt8# (# x3 #)) (packInt8# (# x4 #)) (packInt8# (# x5 #)) (packInt8# (# x6 #)) (packInt8# (# x7 #)) (packInt8# (# x8 #))

{-# INLINE unpackInt8X8 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt8X8 :: Int8X8 -> (Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8)
unpackInt8X8 (Int8X8 m1 m2 m3 m4 m5 m6 m7 m8) = case unpackInt8# m1 of
    (# x1 #) -> case unpackInt8# m2 of
        (# x2 #) -> case unpackInt8# m3 of
            (# x3 #) -> case unpackInt8# m4 of
                (# x4 #) -> case unpackInt8# m5 of
                    (# x5 #) -> case unpackInt8# m6 of
                        (# x6 #) -> case unpackInt8# m7 of
                            (# x7 #) -> case unpackInt8# m8 of
                                (# x8 #) -> (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8)

{-# INLINE unsafeInsertInt8X8 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt8X8 :: Int8X8 -> Int8 -> Int -> Int8X8
unsafeInsertInt8X8 (Int8X8 m1 m2 m3 m4 m5 m6 m7 m8) (I8# y) _i@(I# ip) | _i < 1 = Int8X8 (insertInt8# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8
                                                                       | _i < 2 = Int8X8 m1 (insertInt8# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8
                                                                       | _i < 3 = Int8X8 m1 m2 (insertInt8# m3 y (ip -# 2#)) m4 m5 m6 m7 m8
                                                                       | _i < 4 = Int8X8 m1 m2 m3 (insertInt8# m4 y (ip -# 3#)) m5 m6 m7 m8
                                                                       | _i < 5 = Int8X8 m1 m2 m3 m4 (insertInt8# m5 y (ip -# 4#)) m6 m7 m8
                                                                       | _i < 6 = Int8X8 m1 m2 m3 m4 m5 (insertInt8# m6 y (ip -# 5#)) m7 m8
                                                                       | _i < 7 = Int8X8 m1 m2 m3 m4 m5 m6 (insertInt8# m7 y (ip -# 6#)) m8
                                                                       | otherwise = Int8X8 m1 m2 m3 m4 m5 m6 m7 (insertInt8# m8 y (ip -# 7#))

{-# INLINE[1] mapInt8X8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt8X8 :: (Int8 -> Int8) -> Int8X8 -> Int8X8
mapInt8X8 f = mapInt8X8# (\ x -> case f (I8# x) of { I8# y -> y})

{-# RULES "mapVector abs" mapInt8X8 abs = abs #-}
{-# RULES "mapVector signum" mapInt8X8 signum = signum #-}
{-# RULES "mapVector negate" mapInt8X8 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt8X8 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt8X8 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt8X8 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt8X8 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt8X8 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt8X8 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt8X8 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt8X8 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt8X8 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt8X8# #-}
-- | Unboxed helper function.
mapInt8X8# :: (Int# -> Int#) -> Int8X8 -> Int8X8
mapInt8X8# f = \ v -> case unpackInt8X8 v of
    (I8# x1, I8# x2, I8# x3, I8# x4, I8# x5, I8# x6, I8# x7, I8# x8) -> packInt8X8 (I8# (f x1), I8# (f x2), I8# (f x3), I8# (f x4), I8# (f x5), I8# (f x6), I8# (f x7), I8# (f x8))

{-# INLINE[1] zipInt8X8 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt8X8 :: (Int8 -> Int8 -> Int8) -> Int8X8 -> Int8X8 -> Int8X8
zipInt8X8 f = \ v1 v2 -> case unpackInt8X8 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackInt8X8 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8) -> packInt8X8 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8)

{-# RULES "zipVector +" forall a b . zipInt8X8 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt8X8 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt8X8 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt8X8 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt8X8 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt8X8 #-}
-- | Fold the elements of a vector to a single value
foldInt8X8 :: (Int8 -> Int8 -> Int8) -> Int8X8 -> Int8
foldInt8X8 f' = \ v -> case unpackInt8X8 v of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8
    where f !x !y = f' x y

{-# INLINE plusInt8X8 #-}
-- | Add two vectors element-wise.
plusInt8X8 :: Int8X8 -> Int8X8 -> Int8X8
plusInt8X8 (Int8X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int8X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int8X8 (plusInt8# m1_1 m1_2) (plusInt8# m2_1 m2_2) (plusInt8# m3_1 m3_2) (plusInt8# m4_1 m4_2) (plusInt8# m5_1 m5_2) (plusInt8# m6_1 m6_2) (plusInt8# m7_1 m7_2) (plusInt8# m8_1 m8_2)

{-# INLINE minusInt8X8 #-}
-- | Subtract two vectors element-wise.
minusInt8X8 :: Int8X8 -> Int8X8 -> Int8X8
minusInt8X8 (Int8X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int8X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int8X8 (minusInt8# m1_1 m1_2) (minusInt8# m2_1 m2_2) (minusInt8# m3_1 m3_2) (minusInt8# m4_1 m4_2) (minusInt8# m5_1 m5_2) (minusInt8# m6_1 m6_2) (minusInt8# m7_1 m7_2) (minusInt8# m8_1 m8_2)

{-# INLINE timesInt8X8 #-}
-- | Multiply two vectors element-wise.
timesInt8X8 :: Int8X8 -> Int8X8 -> Int8X8
timesInt8X8 (Int8X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int8X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int8X8 (timesInt8# m1_1 m1_2) (timesInt8# m2_1 m2_2) (timesInt8# m3_1 m3_2) (timesInt8# m4_1 m4_2) (timesInt8# m5_1 m5_2) (timesInt8# m6_1 m6_2) (timesInt8# m7_1 m7_2) (timesInt8# m8_1 m8_2)

{-# INLINE quotInt8X8 #-}
-- | Rounds towards zero element-wise.
quotInt8X8 :: Int8X8 -> Int8X8 -> Int8X8
quotInt8X8 (Int8X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int8X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int8X8 (quotInt8# m1_1 m1_2) (quotInt8# m2_1 m2_2) (quotInt8# m3_1 m3_2) (quotInt8# m4_1 m4_2) (quotInt8# m5_1 m5_2) (quotInt8# m6_1 m6_2) (quotInt8# m7_1 m7_2) (quotInt8# m8_1 m8_2)

{-# INLINE remInt8X8 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt8X8 :: Int8X8 -> Int8X8 -> Int8X8
remInt8X8 (Int8X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Int8X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Int8X8 (remInt8# m1_1 m1_2) (remInt8# m2_1 m2_2) (remInt8# m3_1 m3_2) (remInt8# m4_1 m4_2) (remInt8# m5_1 m5_2) (remInt8# m6_1 m6_2) (remInt8# m7_1 m7_2) (remInt8# m8_1 m8_2)

{-# INLINE negateInt8X8 #-}
-- | Negate element-wise.
negateInt8X8 :: Int8X8 -> Int8X8
negateInt8X8 (Int8X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) = Int8X8 (negateInt8# m1_1) (negateInt8# m2_1) (negateInt8# m3_1) (negateInt8# m4_1) (negateInt8# m5_1) (negateInt8# m6_1) (negateInt8# m7_1) (negateInt8# m8_1)

{-# INLINE indexInt8X8Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt8X8Array :: ByteArray -> Int -> Int8X8
indexInt8X8Array (ByteArray a) (I# i) = Int8X8 (indexInt8Array# a ((i *# 8#) +# 0#)) (indexInt8Array# a ((i *# 8#) +# 1#)) (indexInt8Array# a ((i *# 8#) +# 2#)) (indexInt8Array# a ((i *# 8#) +# 3#)) (indexInt8Array# a ((i *# 8#) +# 4#)) (indexInt8Array# a ((i *# 8#) +# 5#)) (indexInt8Array# a ((i *# 8#) +# 6#)) (indexInt8Array# a ((i *# 8#) +# 7#))

{-# INLINE readInt8X8Array #-}
-- | Read a vector from specified index of the mutable array.
readInt8X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int8X8
readInt8X8Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt8Array# a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt8Array# a ((i *# 8#) +# 1#) s1 of
        (# s2, m2 #) -> case readInt8Array# a ((i *# 8#) +# 2#) s2 of
            (# s3, m3 #) -> case readInt8Array# a ((i *# 8#) +# 3#) s3 of
                (# s4, m4 #) -> case readInt8Array# a ((i *# 8#) +# 4#) s4 of
                    (# s5, m5 #) -> case readInt8Array# a ((i *# 8#) +# 5#) s5 of
                        (# s6, m6 #) -> case readInt8Array# a ((i *# 8#) +# 6#) s6 of
                            (# s7, m7 #) -> case readInt8Array# a ((i *# 8#) +# 7#) s7 of
                                (# s8, m8 #) -> (# s8, Int8X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeInt8X8Array #-}
-- | Write a vector to specified index of mutable array.
writeInt8X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int8X8 -> m ()
writeInt8X8Array (MutableByteArray a) (I# i) (Int8X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeInt8Array# a ((i *# 8#) +# 0#) m1) >> primitive_ (writeInt8Array# a ((i *# 8#) +# 1#) m2) >> primitive_ (writeInt8Array# a ((i *# 8#) +# 2#) m3) >> primitive_ (writeInt8Array# a ((i *# 8#) +# 3#) m4) >> primitive_ (writeInt8Array# a ((i *# 8#) +# 4#) m5) >> primitive_ (writeInt8Array# a ((i *# 8#) +# 5#) m6) >> primitive_ (writeInt8Array# a ((i *# 8#) +# 6#) m7) >> primitive_ (writeInt8Array# a ((i *# 8#) +# 7#) m8)

{-# INLINE indexInt8X8OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt8X8OffAddr :: Addr -> Int -> Int8X8
indexInt8X8OffAddr (Addr a) (I# i) = Int8X8 (indexInt8OffAddr# (plusAddr# a ((i *# 8#) +# 0#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 8#) +# 1#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 8#) +# 2#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 8#) +# 3#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 8#) +# 4#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 8#) +# 5#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 8#) +# 6#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 8#) +# 7#)) 0#)

{-# INLINE readInt8X8OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt8X8OffAddr :: PrimMonad m => Addr -> Int -> m Int8X8
readInt8X8OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 1#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 2#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 3#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 4#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 5#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 6#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 7#) s7 of
                                (# s8, m8 #) -> (# s8, Int8X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeInt8X8OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt8X8OffAddr :: PrimMonad m => Addr -> Int -> Int8X8 -> m ()
writeInt8X8OffAddr (Addr a) (I# i) (Int8X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 8#) +# 0#)) 0# m1) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 8#) +# 1#)) 0# m2) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 8#) +# 2#)) 0# m3) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 8#) +# 3#)) 0# m4) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 8#) +# 4#)) 0# m5) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 8#) +# 5#)) 0# m6) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 8#) +# 6#)) 0# m7) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 8#) +# 7#)) 0# m8)


