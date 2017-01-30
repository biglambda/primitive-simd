{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word8X2 (Word8X2) where

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

-- ** Word8X2
data Word8X2 = Word8X2 Word# Word# deriving Typeable

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

instance Eq Word8X2 where
    a == b = case unpackWord8X2 a of
        (x1, x2) -> case unpackWord8X2 b of
            (y1, y2) -> x1 == y1 && x2 == y2

instance Ord Word8X2 where
    a `compare` b = case unpackWord8X2 a of
        (x1, x2) -> case unpackWord8X2 b of
            (y1, y2) -> x1 `compare` y1 <> x2 `compare` y2

instance Show Word8X2 where
    showsPrec _ a s = case unpackWord8X2 a of
        (x1, x2) -> "Word8X2 (" ++ shows x1 (", " ++ shows x2 (")" ++ s))

instance Num Word8X2 where
    (+) = plusWord8X2
    (-) = minusWord8X2
    (*) = timesWord8X2
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word8X2 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word8X2 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word8X2 where
    type Elem Word8X2 = Word8
    type ElemTuple Word8X2 = (Word8, Word8)
    nullVector         = broadcastVector 0
    vectorSize  _      = 2
    elementSize _      = 1
    broadcastVector    = broadcastWord8X2
    unsafeInsertVector = unsafeInsertWord8X2
    packVector         = packWord8X2
    unpackVector       = unpackWord8X2
    mapVector          = mapWord8X2
    zipVector          = zipWord8X2
    foldVector         = foldWord8X2

instance SIMDIntVector Word8X2 where
    quotVector = quotWord8X2
    remVector  = remWord8X2

instance Prim Word8X2 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord8X2Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord8X2Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord8X2Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord8X2OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord8X2OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord8X2OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word8X2 = V_Word8X2 (PV.Vector Word8X2)
newtype instance UV.MVector s Word8X2 = MV_Word8X2 (PMV.MVector s Word8X2)

instance Vector UV.Vector Word8X2 where
    basicUnsafeFreeze (MV_Word8X2 v) = V_Word8X2 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word8X2 v) = MV_Word8X2 <$> PV.unsafeThaw v
    basicLength (V_Word8X2 v) = PV.length v
    basicUnsafeSlice start len (V_Word8X2 v) = V_Word8X2(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word8X2 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word8X2 m) (V_Word8X2 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word8X2 where
    basicLength (MV_Word8X2 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word8X2 v) = MV_Word8X2(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word8X2 v) (MV_Word8X2 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word8X2 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word8X2 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word8X2 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word8X2 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word8X2

{-# INLINE broadcastWord8X2 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord8X2 :: Word8 -> Word8X2
broadcastWord8X2 (W8# x) = case broadcastWord8# x of
    v -> Word8X2 v v

{-# INLINE packWord8X2 #-}
-- | Pack the elements of a tuple into a vector.
packWord8X2 :: (Word8, Word8) -> Word8X2
packWord8X2 (W8# x1, W8# x2) = Word8X2 (packWord8# (# x1 #)) (packWord8# (# x2 #))

{-# INLINE unpackWord8X2 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord8X2 :: Word8X2 -> (Word8, Word8)
unpackWord8X2 (Word8X2 m1 m2) = case unpackWord8# m1 of
    (# x1 #) -> case unpackWord8# m2 of
        (# x2 #) -> (W8# x1, W8# x2)

{-# INLINE unsafeInsertWord8X2 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord8X2 :: Word8X2 -> Word8 -> Int -> Word8X2
unsafeInsertWord8X2 (Word8X2 m1 m2) (W8# y) _i@(I# ip) | _i < 1 = Word8X2 (insertWord8# m1 y (ip -# 0#)) m2
                                                       | otherwise = Word8X2 m1 (insertWord8# m2 y (ip -# 1#))

{-# INLINE[1] mapWord8X2 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord8X2 :: (Word8 -> Word8) -> Word8X2 -> Word8X2
mapWord8X2 f = mapWord8X2# (\ x -> case f (W8# x) of { W8# y -> y})

{-# RULES "mapVector abs" mapWord8X2 abs = abs #-}
{-# RULES "mapVector signum" mapWord8X2 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord8X2 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord8X2 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord8X2 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord8X2 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord8X2 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord8X2 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord8X2 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord8X2 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord8X2 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord8X2# #-}
-- | Unboxed helper function.
mapWord8X2# :: (Word# -> Word#) -> Word8X2 -> Word8X2
mapWord8X2# f = \ v -> case unpackWord8X2 v of
    (W8# x1, W8# x2) -> packWord8X2 (W8# (f x1), W8# (f x2))

{-# INLINE[1] zipWord8X2 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord8X2 :: (Word8 -> Word8 -> Word8) -> Word8X2 -> Word8X2 -> Word8X2
zipWord8X2 f = \ v1 v2 -> case unpackWord8X2 v1 of
    (x1, x2) -> case unpackWord8X2 v2 of
        (y1, y2) -> packWord8X2 (f x1 y1, f x2 y2)

{-# RULES "zipVector +" forall a b . zipWord8X2 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord8X2 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord8X2 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord8X2 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord8X2 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord8X2 #-}
-- | Fold the elements of a vector to a single value
foldWord8X2 :: (Word8 -> Word8 -> Word8) -> Word8X2 -> Word8
foldWord8X2 f' = \ v -> case unpackWord8X2 v of
    (x1, x2) -> x1 `f` x2
    where f !x !y = f' x y

{-# INLINE plusWord8X2 #-}
-- | Add two vectors element-wise.
plusWord8X2 :: Word8X2 -> Word8X2 -> Word8X2
plusWord8X2 (Word8X2 m1_1 m2_1) (Word8X2 m1_2 m2_2) = Word8X2 (plusWord8# m1_1 m1_2) (plusWord8# m2_1 m2_2)

{-# INLINE minusWord8X2 #-}
-- | Subtract two vectors element-wise.
minusWord8X2 :: Word8X2 -> Word8X2 -> Word8X2
minusWord8X2 (Word8X2 m1_1 m2_1) (Word8X2 m1_2 m2_2) = Word8X2 (minusWord8# m1_1 m1_2) (minusWord8# m2_1 m2_2)

{-# INLINE timesWord8X2 #-}
-- | Multiply two vectors element-wise.
timesWord8X2 :: Word8X2 -> Word8X2 -> Word8X2
timesWord8X2 (Word8X2 m1_1 m2_1) (Word8X2 m1_2 m2_2) = Word8X2 (timesWord8# m1_1 m1_2) (timesWord8# m2_1 m2_2)

{-# INLINE quotWord8X2 #-}
-- | Rounds towards zero element-wise.
quotWord8X2 :: Word8X2 -> Word8X2 -> Word8X2
quotWord8X2 (Word8X2 m1_1 m2_1) (Word8X2 m1_2 m2_2) = Word8X2 (quotWord8# m1_1 m1_2) (quotWord8# m2_1 m2_2)

{-# INLINE remWord8X2 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord8X2 :: Word8X2 -> Word8X2 -> Word8X2
remWord8X2 (Word8X2 m1_1 m2_1) (Word8X2 m1_2 m2_2) = Word8X2 (remWord8# m1_1 m1_2) (remWord8# m2_1 m2_2)

{-# INLINE indexWord8X2Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord8X2Array :: ByteArray -> Int -> Word8X2
indexWord8X2Array (ByteArray a) (I# i) = Word8X2 (indexWord8Array# a ((i *# 2#) +# 0#)) (indexWord8Array# a ((i *# 2#) +# 1#))

{-# INLINE readWord8X2Array #-}
-- | Read a vector from specified index of the mutable array.
readWord8X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word8X2
readWord8X2Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord8Array# a ((i *# 2#) +# 0#) s0 of
    (# s1, m1 #) -> case readWord8Array# a ((i *# 2#) +# 1#) s1 of
        (# s2, m2 #) -> (# s2, Word8X2 m1 m2 #))

{-# INLINE writeWord8X2Array #-}
-- | Write a vector to specified index of mutable array.
writeWord8X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word8X2 -> m ()
writeWord8X2Array (MutableByteArray a) (I# i) (Word8X2 m1 m2) = primitive_ (writeWord8Array# a ((i *# 2#) +# 0#) m1) >> primitive_ (writeWord8Array# a ((i *# 2#) +# 1#) m2)

{-# INLINE indexWord8X2OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord8X2OffAddr :: Addr -> Int -> Word8X2
indexWord8X2OffAddr (Addr a) (I# i) = Word8X2 (indexWord8OffAddr# (plusAddr# a ((i *# 2#) +# 0#)) 0#) (indexWord8OffAddr# (plusAddr# a ((i *# 2#) +# 1#)) 0#)

{-# INLINE readWord8X2OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord8X2OffAddr :: PrimMonad m => Addr -> Int -> m Word8X2
readWord8X2OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 2#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readWord8OffAddr# (plusAddr# addr i') 0#) a ((i *# 2#) +# 1#) s1 of
        (# s2, m2 #) -> (# s2, Word8X2 m1 m2 #))

{-# INLINE writeWord8X2OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord8X2OffAddr :: PrimMonad m => Addr -> Int -> Word8X2 -> m ()
writeWord8X2OffAddr (Addr a) (I# i) (Word8X2 m1 m2) = primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 2#) +# 0#)) 0# m1) >> primitive_ (writeWord8OffAddr# (plusAddr# a ((i *# 2#) +# 1#)) 0# m2)


