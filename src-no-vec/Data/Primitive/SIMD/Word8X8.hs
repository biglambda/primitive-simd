{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word8X8 (Word8X8) where

-- This code was AUTOMATICALLY generated, DO NOT EDIT!

import Data.Primitive.SIMD.Class

import GHC.Word
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

-- ** Word8X8
data Word8X8 = Word8X8 Word# Word# Word# Word# Word# Word# Word# Word# deriving Typeable

broadcastWord8# :: Word# -> Word#
broadcastWord8# v = v

packWord8# :: (# Word# #) -> Word#
packWord8# (# v #) = v

unpackWord8# :: Word# -> (# Word# #)
unpackWord8# v = (# v #)

insertWord8# :: Word# -> Word# -> Int# -> Word#
insertWord8# _ v _ = v

plusWord8# :: Word# -> Word# -> Word#
plusWord8# a b = case W8# a + W8# b of W8# c -> c

minusWord8# :: Word# -> Word# -> Word#
minusWord8# a b = case W8# a - W8# b of W8# c -> c

timesWord8# :: Word# -> Word# -> Word#
timesWord8# a b = case W8# a * W8# b of W8# c -> c

quotWord8# :: Word# -> Word# -> Word#
quotWord8# a b = case W8# a `quot` W8# b of W8# c -> c

remWord8# :: Word# -> Word# -> Word#
remWord8# a b = case W8# a `rem` W8# b of W8# c -> c

abs' :: Word8 -> Word8
abs' (W8# x) = W8# (abs# x)

{-# INLINE abs# #-}
abs# :: Word# -> Word#
abs# x = case abs (W8# x) of
    W8# y -> y

signum' :: Word8 -> Word8
signum' (W8# x) = W8# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Word# -> Word#
signum# x = case signum (W8# x) of
    W8# y -> y

instance Eq Word8X8 where
    a == b = case unpackWord8X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackWord8X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4 && x5 == y5 && x6 == y6 && x7 == y7 && x8 == y8

instance Ord Word8X8 where
    a `compare` b = case unpackWord8X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackWord8X8 b of
            (y1, y2, y3, y4, y5, y6, y7, y8) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4 <> x5 `compare` y5 <> x6 `compare` y6 <> x7 `compare` y7 <> x8 `compare` y8

instance Show Word8X8 where
    showsPrec _ a s = case unpackWord8X8 a of
        (x1, x2, x3, x4, x5, x6, x7, x8) -> "Word8X8 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (", " ++ shows x5 (", " ++ shows x6 (", " ++ shows x7 (", " ++ shows x8 (")" ++ s))))))))

instance Num Word8X8 where
    (+) = plusWord8X8
    (-) = minusWord8X8
    (*) = timesWord8X8
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word8X8 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word8X8 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word8X8 where
    type Elem Word8X8 = Word8
    type ElemTuple Word8X8 = (Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8)
    nullVector         = broadcastVector 0
    vectorSize  _      = 8
    elementSize _      = 1
    broadcastVector    = broadcastWord8X8
    unsafeInsertVector = unsafeInsertWord8X8
    packVector         = packWord8X8
    unpackVector       = unpackWord8X8
    mapVector          = mapWord8X8
    zipVector          = zipWord8X8
    foldVector         = foldWord8X8

instance SIMDIntVector Word8X8 where
    quotVector = quotWord8X8
    remVector  = remWord8X8

instance Prim Word8X8 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord8X8Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord8X8Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord8X8Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord8X8OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord8X8OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord8X8OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word8X8 = V_Word8X8 (PV.Vector Word8X8)
newtype instance UV.MVector s Word8X8 = MV_Word8X8 (PMV.MVector s Word8X8)

instance Vector UV.Vector Word8X8 where
    basicUnsafeFreeze (MV_Word8X8 v) = V_Word8X8 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word8X8 v) = MV_Word8X8 <$> PV.unsafeThaw v
    basicLength (V_Word8X8 v) = PV.length v
    basicUnsafeSlice start len (V_Word8X8 v) = V_Word8X8(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word8X8 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word8X8 m) (V_Word8X8 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word8X8 where
    basicLength (MV_Word8X8 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word8X8 v) = MV_Word8X8(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word8X8 v) (MV_Word8X8 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word8X8 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word8X8 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word8X8 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word8X8 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word8X8

{-# INLINE broadcastWord8X8 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord8X8 :: Word8 -> Word8X8
broadcastWord8X8 (W8# x) = case broadcastWord8# x of
    v -> Word8X8 v v v v v v v v

{-# INLINE packWord8X8 #-}
-- | Pack the elements of a tuple into a vector.
packWord8X8 :: (Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8) -> Word8X8
packWord8X8 (W8# x1, W8# x2, W8# x3, W8# x4, W8# x5, W8# x6, W8# x7, W8# x8) = Word8X8 (packWord8# (# x1 #)) (packWord8# (# x2 #)) (packWord8# (# x3 #)) (packWord8# (# x4 #)) (packWord8# (# x5 #)) (packWord8# (# x6 #)) (packWord8# (# x7 #)) (packWord8# (# x8 #))

{-# INLINE unpackWord8X8 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord8X8 :: Word8X8 -> (Word8, Word8, Word8, Word8, Word8, Word8, Word8, Word8)
unpackWord8X8 (Word8X8 m1 m2 m3 m4 m5 m6 m7 m8) = case unpackWord8# m1 of
    (# x1 #) -> case unpackWord8# m2 of
        (# x2 #) -> case unpackWord8# m3 of
            (# x3 #) -> case unpackWord8# m4 of
                (# x4 #) -> case unpackWord8# m5 of
                    (# x5 #) -> case unpackWord8# m6 of
                        (# x6 #) -> case unpackWord8# m7 of
                            (# x7 #) -> case unpackWord8# m8 of
                                (# x8 #) -> (W8# x1, W8# x2, W8# x3, W8# x4, W8# x5, W8# x6, W8# x7, W8# x8)

{-# INLINE unsafeInsertWord8X8 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord8X8 :: Word8X8 -> Word8 -> Int -> Word8X8
unsafeInsertWord8X8 (Word8X8 m1 m2 m3 m4 m5 m6 m7 m8) (W8# y) _i@(I# ip) | _i < 1 = Word8X8 (insertWord8# m1 y (ip -# 0#)) m2 m3 m4 m5 m6 m7 m8
                                                                         | _i < 2 = Word8X8 m1 (insertWord8# m2 y (ip -# 1#)) m3 m4 m5 m6 m7 m8
                                                                         | _i < 3 = Word8X8 m1 m2 (insertWord8# m3 y (ip -# 2#)) m4 m5 m6 m7 m8
                                                                         | _i < 4 = Word8X8 m1 m2 m3 (insertWord8# m4 y (ip -# 3#)) m5 m6 m7 m8
                                                                         | _i < 5 = Word8X8 m1 m2 m3 m4 (insertWord8# m5 y (ip -# 4#)) m6 m7 m8
                                                                         | _i < 6 = Word8X8 m1 m2 m3 m4 m5 (insertWord8# m6 y (ip -# 5#)) m7 m8
                                                                         | _i < 7 = Word8X8 m1 m2 m3 m4 m5 m6 (insertWord8# m7 y (ip -# 6#)) m8
                                                                         | otherwise = Word8X8 m1 m2 m3 m4 m5 m6 m7 (insertWord8# m8 y (ip -# 7#))

{-# INLINE[1] mapWord8X8 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord8X8 :: (Word8 -> Word8) -> Word8X8 -> Word8X8
mapWord8X8 f = mapWord8X8# (\ x -> case f (W8# x) of { W8# y -> y})

{-# RULES "mapVector abs" mapWord8X8 abs = abs #-}
{-# RULES "mapVector signum" mapWord8X8 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord8X8 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord8X8 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord8X8 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord8X8 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord8X8 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord8X8 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord8X8 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord8X8 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord8X8 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord8X8# #-}
-- | Unboxed helper function.
mapWord8X8# :: (Word# -> Word#) -> Word8X8 -> Word8X8
mapWord8X8# f = \ v -> case unpackWord8X8 v of
    (W8# x1, W8# x2, W8# x3, W8# x4, W8# x5, W8# x6, W8# x7, W8# x8) -> packWord8X8 (W8# (f x1), W8# (f x2), W8# (f x3), W8# (f x4), W8# (f x5), W8# (f x6), W8# (f x7), W8# (f x8))

{-# INLINE[1] zipWord8X8 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord8X8 :: (Word8 -> Word8 -> Word8) -> Word8X8 -> Word8X8 -> Word8X8
zipWord8X8 f = \ v1 v2 -> case unpackWord8X8 v1 of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> case unpackWord8X8 v2 of
        (y1, y2, y3, y4, y5, y6, y7, y8) -> packWord8X8 (f x1 y1, f x2 y2, f x3 y3, f x4 y4, f x5 y5, f x6 y6, f x7 y7, f x8 y8)

{-# RULES "zipVector +" forall a b . zipWord8X8 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord8X8 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord8X8 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord8X8 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord8X8 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord8X8 #-}
-- | Fold the elements of a vector to a single value
foldWord8X8 :: (Word8 -> Word8 -> Word8) -> Word8X8 -> Word8
foldWord8X8 f' = \ v -> case unpackWord8X8 v of
    (x1, x2, x3, x4, x5, x6, x7, x8) -> x1 `f` x2 `f` x3 `f` x4 `f` x5 `f` x6 `f` x7 `f` x8
    where f !x !y = f' x y

{-# INLINE plusWord8X8 #-}
-- | Add two vectors element-wise.
plusWord8X8 :: Word8X8 -> Word8X8 -> Word8X8
plusWord8X8 (Word8X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word8X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word8X8 (plusWord8# m1_1 m1_2) (plusWord8# m2_1 m2_2) (plusWord8# m3_1 m3_2) (plusWord8# m4_1 m4_2) (plusWord8# m5_1 m5_2) (plusWord8# m6_1 m6_2) (plusWord8# m7_1 m7_2) (plusWord8# m8_1 m8_2)

{-# INLINE minusWord8X8 #-}
-- | Subtract two vectors element-wise.
minusWord8X8 :: Word8X8 -> Word8X8 -> Word8X8
minusWord8X8 (Word8X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word8X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word8X8 (minusWord8# m1_1 m1_2) (minusWord8# m2_1 m2_2) (minusWord8# m3_1 m3_2) (minusWord8# m4_1 m4_2) (minusWord8# m5_1 m5_2) (minusWord8# m6_1 m6_2) (minusWord8# m7_1 m7_2) (minusWord8# m8_1 m8_2)

{-# INLINE timesWord8X8 #-}
-- | Multiply two vectors element-wise.
timesWord8X8 :: Word8X8 -> Word8X8 -> Word8X8
timesWord8X8 (Word8X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word8X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word8X8 (timesWord8# m1_1 m1_2) (timesWord8# m2_1 m2_2) (timesWord8# m3_1 m3_2) (timesWord8# m4_1 m4_2) (timesWord8# m5_1 m5_2) (timesWord8# m6_1 m6_2) (timesWord8# m7_1 m7_2) (timesWord8# m8_1 m8_2)

{-# INLINE quotWord8X8 #-}
-- | Rounds towards zero element-wise.
quotWord8X8 :: Word8X8 -> Word8X8 -> Word8X8
quotWord8X8 (Word8X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word8X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word8X8 (quotWord8# m1_1 m1_2) (quotWord8# m2_1 m2_2) (quotWord8# m3_1 m3_2) (quotWord8# m4_1 m4_2) (quotWord8# m5_1 m5_2) (quotWord8# m6_1 m6_2) (quotWord8# m7_1 m7_2) (quotWord8# m8_1 m8_2)

{-# INLINE remWord8X8 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord8X8 :: Word8X8 -> Word8X8 -> Word8X8
remWord8X8 (Word8X8 m1_1 m2_1 m3_1 m4_1 m5_1 m6_1 m7_1 m8_1) (Word8X8 m1_2 m2_2 m3_2 m4_2 m5_2 m6_2 m7_2 m8_2) = Word8X8 (remWord8# m1_1 m1_2) (remWord8# m2_1 m2_2) (remWord8# m3_1 m3_2) (remWord8# m4_1 m4_2) (remWord8# m5_1 m5_2) (remWord8# m6_1 m6_2) (remWord8# m7_1 m7_2) (remWord8# m8_1 m8_2)

{-# INLINE indexWord8X8Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord8X8Array :: ByteArray -> Int -> Word8X8
indexWord8X8Array (ByteArray a) (I# i) = Word8X8 (indexWord8Array# a ((i *# 8#) +# 0#)) (indexWord8Array# a ((i *# 8#) +# 1#)) (indexWord8Array# a ((i *# 8#) +# 2#)) (indexWord8Array# a ((i *# 8#) +# 3#)) (indexWord8Array# a ((i *# 8#) +# 4#)) (indexWord8Array# a ((i *# 8#) +# 5#)) (indexWord8Array# a ((i *# 8#) +# 6#)) (indexWord8Array# a ((i *# 8#) +# 7#))

{-# INLINE readWord8X8Array #-}
-- | Read a vector from specified index of the mutable array.
readWord8X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word8X8
readWord8X8Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord8Array# a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord8Array# a ((i *# 8#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord8Array# a ((i *# 8#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord8Array# a ((i *# 8#) +# 3#) s3 of
                (# s4, m4 #) -> case readWord8Array# a ((i *# 8#) +# 4#) s4 of
                    (# s5, m5 #) -> case readWord8Array# a ((i *# 8#) +# 5#) s5 of
                        (# s6, m6 #) -> case readWord8Array# a ((i *# 8#) +# 6#) s6 of
                            (# s7, m7 #) -> case readWord8Array# a ((i *# 8#) +# 7#) s7 of
                                (# s8, m8 #) -> (# s8, Word8X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeWord8X8Array #-}
-- | Write a vector to specified index of mutable array.
writeWord8X8Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word8X8 -> m ()
writeWord8X8Array (MutableByteArray a) (I# i) (Word8X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeWord8Array# a ((i *# 8#) +# 0#) m1) >> primitive_ (writeWord8Array# a ((i *# 8#) +# 1#) m2) >> primitive_ (writeWord8Array# a ((i *# 8#) +# 2#) m3) >> primitive_ (writeWord8Array# a ((i *# 8#) +# 3#) m4) >> primitive_ (writeWord8Array# a ((i *# 8#) +# 4#) m5) >> primitive_ (writeWord8Array# a ((i *# 8#) +# 5#) m6) >> primitive_ (writeWord8Array# a ((i *# 8#) +# 6#) m7) >> primitive_ (writeWord8Array# a ((i *# 8#) +# 7#) m8)

{-# INLINE indexWord8X8OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord8X8OffAddr :: Addr -> Int -> Word8X8
indexWord8X8OffAddr (Addr a) (I# i) = Word8X8 (indexWord8OffAddr# (plusAddr# a ((i *# 8#) +# 0#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 8#) +# 1#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 8#) +# 2#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 8#) +# 3#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 8#) +# 4#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 8#) +# 5#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 8#) +# 6#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 8#) +# 7#)) 0#)

{-# INLINE readWord8X8OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord8X8OffAddr :: PrimMonad m => Addr -> Int -> m Word8X8
readWord8X8OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 1#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 2#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 3#) s3 of
                (# s4, m4 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 4#) s4 of
                    (# s5, m5 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 5#) s5 of
                        (# s6, m6 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 6#) s6 of
                            (# s7, m7 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 7#) s7 of
                                (# s8, m8 #) -> (# s8, Word8X8 m1 m2 m3 m4 m5 m6 m7 m8 #))

{-# INLINE writeWord8X8OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord8X8OffAddr :: PrimMonad m => Addr -> Int -> Word8X8 -> m ()
writeWord8X8OffAddr (Addr a) (I# i) (Word8X8 m1 m2 m3 m4 m5 m6 m7 m8) = primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 8#) +# 0#)) 0# m1) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 8#) +# 1#)) 0# m2) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 8#) +# 2#)) 0# m3) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 8#) +# 3#)) 0# m4) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 8#) +# 4#)) 0# m5) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 8#) +# 5#)) 0# m6) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 8#) +# 6#)) 0# m7) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 8#) +# 7#)) 0# m8)


