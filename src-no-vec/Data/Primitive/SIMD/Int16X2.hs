{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int16X2 (Int16X2) where

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

-- ** Int16X2
data Int16X2 = Int16X2 Int# Int# deriving Typeable

broadcastInt16# :: Int# -> Int#
broadcastInt16# v = v

packInt16# :: (# Int# #) -> Int#
packInt16# (# v #) = v

unpackInt16# :: Int# -> (# Int# #)
unpackInt16# v = (# v #)

insertInt16# :: Int# -> Int# -> Int# -> Int#
insertInt16# _ v _ = v

negateInt16# :: Int# -> Int#
negateInt16# a = case negate (I16# a) of I16# b -> b

plusInt16# :: Int# -> Int# -> Int#
plusInt16# a b = case I16# a + I16# b of I16# c -> c

minusInt16# :: Int# -> Int# -> Int#
minusInt16# a b = case I16# a - I16# b of I16# c -> c

timesInt16# :: Int# -> Int# -> Int#
timesInt16# a b = case I16# a * I16# b of I16# c -> c

quotInt16# :: Int# -> Int# -> Int#
quotInt16# a b = case I16# a `quot` I16# b of I16# c -> c

remInt16# :: Int# -> Int# -> Int#
remInt16# a b = case I16# a `rem` I16# b of I16# c -> c

abs' :: Int16 -> Int16
abs' (I16# x) = I16# (abs# x)

{-# NOINLINE abs# #-}
abs# :: Int# -> Int#
abs# x = case abs (I16# x) of
    I16# y -> y

signum' :: Int16 -> Int16
signum' (I16# x) = I16# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Int# -> Int#
signum# x = case signum (I16# x) of
    I16# y -> y

instance Eq Int16X2 where
    a == b = case unpackInt16X2 a of
        (x1, x2) -> case unpackInt16X2 b of
            (y1, y2) -> x1 == y1 && x2 == y2

instance Ord Int16X2 where
    a `compare` b = case unpackInt16X2 a of
        (x1, x2) -> case unpackInt16X2 b of
            (y1, y2) -> x1 `compare` y1 <> x2 `compare` y2

instance Show Int16X2 where
    showsPrec _ a s = case unpackInt16X2 a of
        (x1, x2) -> "Int16X2 (" ++ shows x1 (", " ++ shows x2 (")" ++ s))

instance Num Int16X2 where
    (+) = plusInt16X2
    (-) = minusInt16X2
    (*) = timesInt16X2
    negate = negateInt16X2
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int16X2 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int16X2 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int16X2 where
    type Elem Int16X2 = Int16
    type ElemTuple Int16X2 = (Int16, Int16)
    nullVector         = broadcastVector 0
    vectorSize  _      = 2
    elementSize _      = 2
    broadcastVector    = broadcastInt16X2
    unsafeInsertVector = unsafeInsertInt16X2
    packVector         = packInt16X2
    unpackVector       = unpackInt16X2
    mapVector          = mapInt16X2
    zipVector          = zipInt16X2
    foldVector         = foldInt16X2

instance SIMDIntVector Int16X2 where
    quotVector = quotInt16X2
    remVector  = remInt16X2

instance Prim Int16X2 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt16X2Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt16X2Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt16X2Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt16X2OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt16X2OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt16X2OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int16X2 = V_Int16X2 (PV.Vector Int16X2)
newtype instance UV.MVector s Int16X2 = MV_Int16X2 (PMV.MVector s Int16X2)

instance Vector UV.Vector Int16X2 where
    basicUnsafeFreeze (MV_Int16X2 v) = V_Int16X2 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int16X2 v) = MV_Int16X2 <$> PV.unsafeThaw v
    basicLength (V_Int16X2 v) = PV.length v
    basicUnsafeSlice start len (V_Int16X2 v) = V_Int16X2(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int16X2 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int16X2 m) (V_Int16X2 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int16X2 where
    basicLength (MV_Int16X2 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int16X2 v) = MV_Int16X2(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int16X2 v) (MV_Int16X2 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int16X2 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int16X2 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int16X2 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int16X2 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int16X2

{-# INLINE broadcastInt16X2 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt16X2 :: Int16 -> Int16X2
broadcastInt16X2 (I16# x) = case broadcastInt16# x of
    v -> Int16X2 v v

{-# INLINE packInt16X2 #-}
-- | Pack the elements of a tuple into a vector.
packInt16X2 :: (Int16, Int16) -> Int16X2
packInt16X2 (I16# x1, I16# x2) = Int16X2 (packInt16# (# x1 #)) (packInt16# (# x2 #))

{-# INLINE unpackInt16X2 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt16X2 :: Int16X2 -> (Int16, Int16)
unpackInt16X2 (Int16X2 m1 m2) = case unpackInt16# m1 of
    (# x1 #) -> case unpackInt16# m2 of
        (# x2 #) -> (I16# x1, I16# x2)

{-# INLINE unsafeInsertInt16X2 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt16X2 :: Int16X2 -> Int16 -> Int -> Int16X2
unsafeInsertInt16X2 (Int16X2 m1 m2) (I16# y) _i@(I# ip) | _i < 1 = Int16X2 (insertInt16# m1 y (ip -# 0#)) m2
                                                        | otherwise = Int16X2 m1 (insertInt16# m2 y (ip -# 1#))

{-# INLINE[1] mapInt16X2 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt16X2 :: (Int16 -> Int16) -> Int16X2 -> Int16X2
mapInt16X2 f = mapInt16X2# (\ x -> case f (I16# x) of { I16# y -> y})

{-# RULES "mapVector abs" mapInt16X2 abs = abs #-}
{-# RULES "mapVector signum" mapInt16X2 signum = signum #-}
{-# RULES "mapVector negate" mapInt16X2 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt16X2 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt16X2 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt16X2 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt16X2 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt16X2 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt16X2 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt16X2 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt16X2 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt16X2 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt16X2# #-}
-- | Unboxed helper function.
mapInt16X2# :: (Int# -> Int#) -> Int16X2 -> Int16X2
mapInt16X2# f = \ v -> case unpackInt16X2 v of
    (I16# x1, I16# x2) -> packInt16X2 (I16# (f x1), I16# (f x2))

{-# INLINE[1] zipInt16X2 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt16X2 :: (Int16 -> Int16 -> Int16) -> Int16X2 -> Int16X2 -> Int16X2
zipInt16X2 f = \ v1 v2 -> case unpackInt16X2 v1 of
    (x1, x2) -> case unpackInt16X2 v2 of
        (y1, y2) -> packInt16X2 (f x1 y1, f x2 y2)

{-# RULES "zipVector +" forall a b . zipInt16X2 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt16X2 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt16X2 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt16X2 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt16X2 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt16X2 #-}
-- | Fold the elements of a vector to a single value
foldInt16X2 :: (Int16 -> Int16 -> Int16) -> Int16X2 -> Int16
foldInt16X2 f' = \ v -> case unpackInt16X2 v of
    (x1, x2) -> x1 `f` x2
    where f !x !y = f' x y

{-# INLINE plusInt16X2 #-}
-- | Add two vectors element-wise.
plusInt16X2 :: Int16X2 -> Int16X2 -> Int16X2
plusInt16X2 (Int16X2 m1_1 m2_1) (Int16X2 m1_2 m2_2) = Int16X2 (plusInt16# m1_1 m1_2) (plusInt16# m2_1 m2_2)

{-# INLINE minusInt16X2 #-}
-- | Subtract two vectors element-wise.
minusInt16X2 :: Int16X2 -> Int16X2 -> Int16X2
minusInt16X2 (Int16X2 m1_1 m2_1) (Int16X2 m1_2 m2_2) = Int16X2 (minusInt16# m1_1 m1_2) (minusInt16# m2_1 m2_2)

{-# INLINE timesInt16X2 #-}
-- | Multiply two vectors element-wise.
timesInt16X2 :: Int16X2 -> Int16X2 -> Int16X2
timesInt16X2 (Int16X2 m1_1 m2_1) (Int16X2 m1_2 m2_2) = Int16X2 (timesInt16# m1_1 m1_2) (timesInt16# m2_1 m2_2)

{-# INLINE quotInt16X2 #-}
-- | Rounds towards zero element-wise.
quotInt16X2 :: Int16X2 -> Int16X2 -> Int16X2
quotInt16X2 (Int16X2 m1_1 m2_1) (Int16X2 m1_2 m2_2) = Int16X2 (quotInt16# m1_1 m1_2) (quotInt16# m2_1 m2_2)

{-# INLINE remInt16X2 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt16X2 :: Int16X2 -> Int16X2 -> Int16X2
remInt16X2 (Int16X2 m1_1 m2_1) (Int16X2 m1_2 m2_2) = Int16X2 (remInt16# m1_1 m1_2) (remInt16# m2_1 m2_2)

{-# INLINE negateInt16X2 #-}
-- | Negate element-wise.
negateInt16X2 :: Int16X2 -> Int16X2
negateInt16X2 (Int16X2 m1_1 m2_1) = Int16X2 (negateInt16# m1_1) (negateInt16# m2_1)

{-# INLINE indexInt16X2Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt16X2Array :: ByteArray -> Int -> Int16X2
indexInt16X2Array (ByteArray a) (I# i) = Int16X2 (indexInt16Array# a ((i *# 2#) +# 0#)) (indexInt16Array# a ((i *# 2#) +# 1#))

{-# INLINE readInt16X2Array #-}
-- | Read a vector from specified index of the mutable array.
readInt16X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int16X2
readInt16X2Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt16Array# a ((i *# 2#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt16Array# a ((i *# 2#) +# 1#) s1 of
        (# s2, m2 #) -> (# s2, Int16X2 m1 m2 #))

{-# INLINE writeInt16X2Array #-}
-- | Write a vector to specified index of mutable array.
writeInt16X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int16X2 -> m ()
writeInt16X2Array (MutableByteArray a) (I# i) (Int16X2 m1 m2) = primitive_ (writeInt16Array# a ((i *# 2#) +# 0#) m1) >> primitive_ (writeInt16Array# a ((i *# 2#) +# 1#) m2)

{-# INLINE indexInt16X2OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt16X2OffAddr :: Addr -> Int -> Int16X2
indexInt16X2OffAddr (Addr a) (I# i) = Int16X2 (indexInt16OffAddr# (plusAddr# a ((i *# 4#) +# 0#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 4#) +# 2#)) 0#)

{-# INLINE readInt16X2OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt16X2OffAddr :: PrimMonad m => Addr -> Int -> m Int16X2
readInt16X2OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 4#) +# 2#) s1 of
        (# s2, m2 #) -> (# s2, Int16X2 m1 m2 #))

{-# INLINE writeInt16X2OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt16X2OffAddr :: PrimMonad m => Addr -> Int -> Int16X2 -> m ()
writeInt16X2OffAddr (Addr a) (I# i) (Int16X2 m1 m2) = primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 4#) +# 0#)) 0# m1) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 4#) +# 2#)) 0# m2)


