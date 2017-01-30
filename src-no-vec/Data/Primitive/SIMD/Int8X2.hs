{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int8X2 (Int8X2) where

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

-- ** Int8X2
data Int8X2 = Int8X2 Int# Int# deriving Typeable

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

instance Eq Int8X2 where
    a == b = case unpackInt8X2 a of
        (x1, x2) -> case unpackInt8X2 b of
            (y1, y2) -> x1 == y1 && x2 == y2

instance Ord Int8X2 where
    a `compare` b = case unpackInt8X2 a of
        (x1, x2) -> case unpackInt8X2 b of
            (y1, y2) -> x1 `compare` y1 <> x2 `compare` y2

instance Show Int8X2 where
    showsPrec _ a s = case unpackInt8X2 a of
        (x1, x2) -> "Int8X2 (" ++ shows x1 (", " ++ shows x2 (")" ++ s))

instance Num Int8X2 where
    (+) = plusInt8X2
    (-) = minusInt8X2
    (*) = timesInt8X2
    negate = negateInt8X2
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int8X2 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int8X2 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int8X2 where
    type Elem Int8X2 = Int8
    type ElemTuple Int8X2 = (Int8, Int8)
    nullVector         = broadcastVector 0
    vectorSize  _      = 2
    elementSize _      = 1
    broadcastVector    = broadcastInt8X2
    unsafeInsertVector = unsafeInsertInt8X2
    packVector         = packInt8X2
    unpackVector       = unpackInt8X2
    mapVector          = mapInt8X2
    zipVector          = zipInt8X2
    foldVector         = foldInt8X2

instance SIMDIntVector Int8X2 where
    quotVector = quotInt8X2
    remVector  = remInt8X2

instance Prim Int8X2 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt8X2Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt8X2Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt8X2Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt8X2OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt8X2OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt8X2OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int8X2 = V_Int8X2 (PV.Vector Int8X2)
newtype instance UV.MVector s Int8X2 = MV_Int8X2 (PMV.MVector s Int8X2)

instance Vector UV.Vector Int8X2 where
    basicUnsafeFreeze (MV_Int8X2 v) = V_Int8X2 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int8X2 v) = MV_Int8X2 <$> PV.unsafeThaw v
    basicLength (V_Int8X2 v) = PV.length v
    basicUnsafeSlice start len (V_Int8X2 v) = V_Int8X2(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int8X2 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int8X2 m) (V_Int8X2 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int8X2 where
    basicLength (MV_Int8X2 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int8X2 v) = MV_Int8X2(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int8X2 v) (MV_Int8X2 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int8X2 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int8X2 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int8X2 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int8X2 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int8X2

{-# INLINE broadcastInt8X2 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt8X2 :: Int8 -> Int8X2
broadcastInt8X2 (I8# x) = case broadcastInt8# x of
    v -> Int8X2 v v

{-# INLINE packInt8X2 #-}
-- | Pack the elements of a tuple into a vector.
packInt8X2 :: (Int8, Int8) -> Int8X2
packInt8X2 (I8# x1, I8# x2) = Int8X2 (packInt8# (# x1 #)) (packInt8# (# x2 #))

{-# INLINE unpackInt8X2 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt8X2 :: Int8X2 -> (Int8, Int8)
unpackInt8X2 (Int8X2 m1 m2) = case unpackInt8# m1 of
    (# x1 #) -> case unpackInt8# m2 of
        (# x2 #) -> (I8# x1, I8# x2)

{-# INLINE unsafeInsertInt8X2 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt8X2 :: Int8X2 -> Int8 -> Int -> Int8X2
unsafeInsertInt8X2 (Int8X2 m1 m2) (I8# y) _i@(I# ip) | _i < 1 = Int8X2 (insertInt8# m1 y (ip -# 0#)) m2
                                                     | otherwise = Int8X2 m1 (insertInt8# m2 y (ip -# 1#))

{-# INLINE[1] mapInt8X2 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt8X2 :: (Int8 -> Int8) -> Int8X2 -> Int8X2
mapInt8X2 f = mapInt8X2# (\ x -> case f (I8# x) of { I8# y -> y})

{-# RULES "mapVector abs" mapInt8X2 abs = abs #-}
{-# RULES "mapVector signum" mapInt8X2 signum = signum #-}
{-# RULES "mapVector negate" mapInt8X2 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt8X2 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt8X2 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt8X2 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt8X2 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt8X2 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt8X2 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt8X2 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt8X2 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt8X2 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt8X2# #-}
-- | Unboxed helper function.
mapInt8X2# :: (Int# -> Int#) -> Int8X2 -> Int8X2
mapInt8X2# f = \ v -> case unpackInt8X2 v of
    (I8# x1, I8# x2) -> packInt8X2 (I8# (f x1), I8# (f x2))

{-# INLINE[1] zipInt8X2 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt8X2 :: (Int8 -> Int8 -> Int8) -> Int8X2 -> Int8X2 -> Int8X2
zipInt8X2 f = \ v1 v2 -> case unpackInt8X2 v1 of
    (x1, x2) -> case unpackInt8X2 v2 of
        (y1, y2) -> packInt8X2 (f x1 y1, f x2 y2)

{-# RULES "zipVector +" forall a b . zipInt8X2 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt8X2 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt8X2 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt8X2 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt8X2 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt8X2 #-}
-- | Fold the elements of a vector to a single value
foldInt8X2 :: (Int8 -> Int8 -> Int8) -> Int8X2 -> Int8
foldInt8X2 f' = \ v -> case unpackInt8X2 v of
    (x1, x2) -> x1 `f` x2
    where f !x !y = f' x y

{-# INLINE plusInt8X2 #-}
-- | Add two vectors element-wise.
plusInt8X2 :: Int8X2 -> Int8X2 -> Int8X2
plusInt8X2 (Int8X2 m1_1 m2_1) (Int8X2 m1_2 m2_2) = Int8X2 (plusInt8# m1_1 m1_2) (plusInt8# m2_1 m2_2)

{-# INLINE minusInt8X2 #-}
-- | Subtract two vectors element-wise.
minusInt8X2 :: Int8X2 -> Int8X2 -> Int8X2
minusInt8X2 (Int8X2 m1_1 m2_1) (Int8X2 m1_2 m2_2) = Int8X2 (minusInt8# m1_1 m1_2) (minusInt8# m2_1 m2_2)

{-# INLINE timesInt8X2 #-}
-- | Multiply two vectors element-wise.
timesInt8X2 :: Int8X2 -> Int8X2 -> Int8X2
timesInt8X2 (Int8X2 m1_1 m2_1) (Int8X2 m1_2 m2_2) = Int8X2 (timesInt8# m1_1 m1_2) (timesInt8# m2_1 m2_2)

{-# INLINE quotInt8X2 #-}
-- | Rounds towards zero element-wise.
quotInt8X2 :: Int8X2 -> Int8X2 -> Int8X2
quotInt8X2 (Int8X2 m1_1 m2_1) (Int8X2 m1_2 m2_2) = Int8X2 (quotInt8# m1_1 m1_2) (quotInt8# m2_1 m2_2)

{-# INLINE remInt8X2 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt8X2 :: Int8X2 -> Int8X2 -> Int8X2
remInt8X2 (Int8X2 m1_1 m2_1) (Int8X2 m1_2 m2_2) = Int8X2 (remInt8# m1_1 m1_2) (remInt8# m2_1 m2_2)

{-# INLINE negateInt8X2 #-}
-- | Negate element-wise.
negateInt8X2 :: Int8X2 -> Int8X2
negateInt8X2 (Int8X2 m1_1 m2_1) = Int8X2 (negateInt8# m1_1) (negateInt8# m2_1)

{-# INLINE indexInt8X2Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt8X2Array :: ByteArray -> Int -> Int8X2
indexInt8X2Array (ByteArray a) (I# i) = Int8X2 (indexInt8Array# a ((i *# 2#) +# 0#)) (indexInt8Array# a ((i *# 2#) +# 1#))

{-# INLINE readInt8X2Array #-}
-- | Read a vector from specified index of the mutable array.
readInt8X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int8X2
readInt8X2Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt8Array# a ((i *# 2#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt8Array# a ((i *# 2#) +# 1#) s1 of
        (# s2, m2 #) -> (# s2, Int8X2 m1 m2 #))

{-# INLINE writeInt8X2Array #-}
-- | Write a vector to specified index of mutable array.
writeInt8X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int8X2 -> m ()
writeInt8X2Array (MutableByteArray a) (I# i) (Int8X2 m1 m2) = primitive_ (writeInt8Array# a ((i *# 2#) +# 0#) m1) >> primitive_ (writeInt8Array# a ((i *# 2#) +# 1#) m2)

{-# INLINE indexInt8X2OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt8X2OffAddr :: Addr -> Int -> Int8X2
indexInt8X2OffAddr (Addr a) (I# i) = Int8X2 (indexInt8OffAddr# (plusAddr# a ((i *# 2#) +# 0#)) 0#) (indexInt8OffAddr# (plusAddr# a ((i *# 2#) +# 1#)) 0#)

{-# INLINE readInt8X2OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt8X2OffAddr :: PrimMonad m => Addr -> Int -> m Int8X2
readInt8X2OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 2#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt8OffAddr# (plusAddr# addr i') 0#) a ((i *# 2#) +# 1#) s1 of
        (# s2, m2 #) -> (# s2, Int8X2 m1 m2 #))

{-# INLINE writeInt8X2OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt8X2OffAddr :: PrimMonad m => Addr -> Int -> Int8X2 -> m ()
writeInt8X2OffAddr (Addr a) (I# i) (Int8X2 m1 m2) = primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 2#) +# 0#)) 0# m1) >> primitive_ (writeInt8OffAddr# (plusAddr# a ((i *# 2#) +# 1#)) 0# m2)


