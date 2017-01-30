{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word16X4 (Word16X4) where

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

-- ** Word16X4
data Word16X4 = Word16X4 Word# Word# Word# Word# deriving Typeable

broadcastWord16# :: Word# -> Word#
broadcastWord16# v = v

packWord16# :: (# Word# #) -> Word#
packWord16# (# v #) = v

unpackWord16# :: Word# -> (# Word# #)
unpackWord16# v = (# v #)

insertWord16# :: Word# -> Word# -> Int# -> Word#
insertWord16# _ v _ = v

plusWord16# :: Word# -> Word# -> Word#
plusWord16# a b = case W16# a + W16# b of W16# c -> c

minusWord16# :: Word# -> Word# -> Word#
minusWord16# a b = case W16# a - W16# b of W16# c -> c

timesWord16# :: Word# -> Word# -> Word#
timesWord16# a b = case W16# a * W16# b of W16# c -> c

quotWord16# :: Word# -> Word# -> Word#
quotWord16# a b = case W16# a `quot` W16# b of W16# c -> c

remWord16# :: Word# -> Word# -> Word#
remWord16# a b = case W16# a `rem` W16# b of W16# c -> c

abs' :: Word16 -> Word16
abs' (W16# x) = W16# (abs# x)

{-# INLINE abs# #-}
abs# :: Word# -> Word#
abs# x = case abs (W16# x) of
    W16# y -> y

signum' :: Word16 -> Word16
signum' (W16# x) = W16# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Word# -> Word#
signum# x = case signum (W16# x) of
    W16# y -> y

instance Eq Word16X4 where
    a == b = case unpackWord16X4 a of
        (x1, x2, x3, x4) -> case unpackWord16X4 b of
            (y1, y2, y3, y4) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4

instance Ord Word16X4 where
    a `compare` b = case unpackWord16X4 a of
        (x1, x2, x3, x4) -> case unpackWord16X4 b of
            (y1, y2, y3, y4) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4

instance Show Word16X4 where
    showsPrec _ a s = case unpackWord16X4 a of
        (x1, x2, x3, x4) -> "Word16X4 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (")" ++ s))))

instance Num Word16X4 where
    (+) = plusWord16X4
    (-) = minusWord16X4
    (*) = timesWord16X4
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word16X4 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word16X4 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word16X4 where
    type Elem Word16X4 = Word16
    type ElemTuple Word16X4 = (Word16, Word16, Word16, Word16)
    nullVector         = broadcastVector 0
    vectorSize  _      = 4
    elementSize _      = 2
    broadcastVector    = broadcastWord16X4
    unsafeInsertVector = unsafeInsertWord16X4
    packVector         = packWord16X4
    unpackVector       = unpackWord16X4
    mapVector          = mapWord16X4
    zipVector          = zipWord16X4
    foldVector         = foldWord16X4

instance SIMDIntVector Word16X4 where
    quotVector = quotWord16X4
    remVector  = remWord16X4

instance Prim Word16X4 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord16X4Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord16X4Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord16X4Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord16X4OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord16X4OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord16X4OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word16X4 = V_Word16X4 (PV.Vector Word16X4)
newtype instance UV.MVector s Word16X4 = MV_Word16X4 (PMV.MVector s Word16X4)

instance Vector UV.Vector Word16X4 where
    basicUnsafeFreeze (MV_Word16X4 v) = V_Word16X4 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word16X4 v) = MV_Word16X4 <$> PV.unsafeThaw v
    basicLength (V_Word16X4 v) = PV.length v
    basicUnsafeSlice start len (V_Word16X4 v) = V_Word16X4(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word16X4 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word16X4 m) (V_Word16X4 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word16X4 where
    basicLength (MV_Word16X4 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word16X4 v) = MV_Word16X4(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word16X4 v) (MV_Word16X4 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word16X4 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word16X4 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word16X4 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word16X4 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word16X4

{-# INLINE broadcastWord16X4 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord16X4 :: Word16 -> Word16X4
broadcastWord16X4 (W16# x) = case broadcastWord16# x of
    v -> Word16X4 v v v v

{-# INLINE packWord16X4 #-}
-- | Pack the elements of a tuple into a vector.
packWord16X4 :: (Word16, Word16, Word16, Word16) -> Word16X4
packWord16X4 (W16# x1, W16# x2, W16# x3, W16# x4) = Word16X4 (packWord16# (# x1 #)) (packWord16# (# x2 #)) (packWord16# (# x3 #)) (packWord16# (# x4 #))

{-# INLINE unpackWord16X4 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord16X4 :: Word16X4 -> (Word16, Word16, Word16, Word16)
unpackWord16X4 (Word16X4 m1 m2 m3 m4) = case unpackWord16# m1 of
    (# x1 #) -> case unpackWord16# m2 of
        (# x2 #) -> case unpackWord16# m3 of
            (# x3 #) -> case unpackWord16# m4 of
                (# x4 #) -> (W16# x1, W16# x2, W16# x3, W16# x4)

{-# INLINE unsafeInsertWord16X4 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord16X4 :: Word16X4 -> Word16 -> Int -> Word16X4
unsafeInsertWord16X4 (Word16X4 m1 m2 m3 m4) (W16# y) _i@(I# ip) | _i < 1 = Word16X4 (insertWord16# m1 y (ip -# 0#)) m2 m3 m4
                                                                | _i < 2 = Word16X4 m1 (insertWord16# m2 y (ip -# 1#)) m3 m4
                                                                | _i < 3 = Word16X4 m1 m2 (insertWord16# m3 y (ip -# 2#)) m4
                                                                | otherwise = Word16X4 m1 m2 m3 (insertWord16# m4 y (ip -# 3#))

{-# INLINE[1] mapWord16X4 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord16X4 :: (Word16 -> Word16) -> Word16X4 -> Word16X4
mapWord16X4 f = mapWord16X4# (\ x -> case f (W16# x) of { W16# y -> y})

{-# RULES "mapVector abs" mapWord16X4 abs = abs #-}
{-# RULES "mapVector signum" mapWord16X4 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord16X4 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord16X4 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord16X4 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord16X4 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord16X4 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord16X4 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord16X4 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord16X4 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord16X4 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord16X4# #-}
-- | Unboxed helper function.
mapWord16X4# :: (Word# -> Word#) -> Word16X4 -> Word16X4
mapWord16X4# f = \ v -> case unpackWord16X4 v of
    (W16# x1, W16# x2, W16# x3, W16# x4) -> packWord16X4 (W16# (f x1), W16# (f x2), W16# (f x3), W16# (f x4))

{-# INLINE[1] zipWord16X4 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord16X4 :: (Word16 -> Word16 -> Word16) -> Word16X4 -> Word16X4 -> Word16X4
zipWord16X4 f = \ v1 v2 -> case unpackWord16X4 v1 of
    (x1, x2, x3, x4) -> case unpackWord16X4 v2 of
        (y1, y2, y3, y4) -> packWord16X4 (f x1 y1, f x2 y2, f x3 y3, f x4 y4)

{-# RULES "zipVector +" forall a b . zipWord16X4 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord16X4 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord16X4 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord16X4 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord16X4 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord16X4 #-}
-- | Fold the elements of a vector to a single value
foldWord16X4 :: (Word16 -> Word16 -> Word16) -> Word16X4 -> Word16
foldWord16X4 f' = \ v -> case unpackWord16X4 v of
    (x1, x2, x3, x4) -> x1 `f` x2 `f` x3 `f` x4
    where f !x !y = f' x y

{-# INLINE plusWord16X4 #-}
-- | Add two vectors element-wise.
plusWord16X4 :: Word16X4 -> Word16X4 -> Word16X4
plusWord16X4 (Word16X4 m1_1 m2_1 m3_1 m4_1) (Word16X4 m1_2 m2_2 m3_2 m4_2) = Word16X4 (plusWord16# m1_1 m1_2) (plusWord16# m2_1 m2_2) (plusWord16# m3_1 m3_2) (plusWord16# m4_1 m4_2)

{-# INLINE minusWord16X4 #-}
-- | Subtract two vectors element-wise.
minusWord16X4 :: Word16X4 -> Word16X4 -> Word16X4
minusWord16X4 (Word16X4 m1_1 m2_1 m3_1 m4_1) (Word16X4 m1_2 m2_2 m3_2 m4_2) = Word16X4 (minusWord16# m1_1 m1_2) (minusWord16# m2_1 m2_2) (minusWord16# m3_1 m3_2) (minusWord16# m4_1 m4_2)

{-# INLINE timesWord16X4 #-}
-- | Multiply two vectors element-wise.
timesWord16X4 :: Word16X4 -> Word16X4 -> Word16X4
timesWord16X4 (Word16X4 m1_1 m2_1 m3_1 m4_1) (Word16X4 m1_2 m2_2 m3_2 m4_2) = Word16X4 (timesWord16# m1_1 m1_2) (timesWord16# m2_1 m2_2) (timesWord16# m3_1 m3_2) (timesWord16# m4_1 m4_2)

{-# INLINE quotWord16X4 #-}
-- | Rounds towards zero element-wise.
quotWord16X4 :: Word16X4 -> Word16X4 -> Word16X4
quotWord16X4 (Word16X4 m1_1 m2_1 m3_1 m4_1) (Word16X4 m1_2 m2_2 m3_2 m4_2) = Word16X4 (quotWord16# m1_1 m1_2) (quotWord16# m2_1 m2_2) (quotWord16# m3_1 m3_2) (quotWord16# m4_1 m4_2)

{-# INLINE remWord16X4 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord16X4 :: Word16X4 -> Word16X4 -> Word16X4
remWord16X4 (Word16X4 m1_1 m2_1 m3_1 m4_1) (Word16X4 m1_2 m2_2 m3_2 m4_2) = Word16X4 (remWord16# m1_1 m1_2) (remWord16# m2_1 m2_2) (remWord16# m3_1 m3_2) (remWord16# m4_1 m4_2)

{-# INLINE indexWord16X4Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord16X4Array :: ByteArray -> Int -> Word16X4
indexWord16X4Array (ByteArray a) (I# i) = Word16X4 (indexWord16Array# a ((i *# 4#) +# 0#)) (indexWord16Array# a ((i *# 4#) +# 1#)) (indexWord16Array# a ((i *# 4#) +# 2#)) (indexWord16Array# a ((i *# 4#) +# 3#))

{-# INLINE readWord16X4Array #-}
-- | Read a vector from specified index of the mutable array.
readWord16X4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word16X4
readWord16X4Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord16Array# a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord16Array# a ((i *# 4#) +# 1#) s1 of
        (# s2, m2 #) -> case readWord16Array# a ((i *# 4#) +# 2#) s2 of
            (# s3, m3 #) -> case readWord16Array# a ((i *# 4#) +# 3#) s3 of
                (# s4, m4 #) -> (# s4, Word16X4 m1 m2 m3 m4 #))

{-# INLINE writeWord16X4Array #-}
-- | Write a vector to specified index of mutable array.
writeWord16X4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word16X4 -> m ()
writeWord16X4Array (MutableByteArray a) (I# i) (Word16X4 m1 m2 m3 m4) = primitive_ (writeWord16Array# a ((i *# 4#) +# 0#) m1) >> primitive_ (writeWord16Array# a ((i *# 4#) +# 1#) m2) >> primitive_ (writeWord16Array# a ((i *# 4#) +# 2#) m3) >> primitive_ (writeWord16Array# a ((i *# 4#) +# 3#) m4)

{-# INLINE indexWord16X4OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord16X4OffAddr :: Addr -> Int -> Word16X4
indexWord16X4OffAddr (Addr a) (I# i) = Word16X4 (indexWord16OffAddr# (plusAddr# a ((i *# 8#) +# 0#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 8#) +# 2#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 8#) +# 4#)) 0#) (indexWord16OffAddr# (plusAddr# a ((i *# 8#) +# 6#)) 0#)

{-# INLINE readWord16X4OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord16X4OffAddr :: PrimMonad m => Addr -> Int -> m Word16X4
readWord16X4OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 2#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 4#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readWord16OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 6#) s3 of
                (# s4, m4 #) -> (# s4, Word16X4 m1 m2 m3 m4 #))

{-# INLINE writeWord16X4OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord16X4OffAddr :: PrimMonad m => Addr -> Int -> Word16X4 -> m ()
writeWord16X4OffAddr (Addr a) (I# i) (Word16X4 m1 m2 m3 m4) = primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 8#) +# 0#)) 0# m1) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 8#) +# 2#)) 0# m2) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 8#) +# 4#)) 0# m3) >> primitive_ (writeWord16OffAddr# (plusAddr# a ((i *# 8#) +# 6#)) 0# m4)


