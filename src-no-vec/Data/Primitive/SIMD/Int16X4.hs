{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int16X4 (Int16X4) where

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

-- ** Int16X4
data Int16X4 = Int16X4 Int# Int# Int# Int# deriving Typeable

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

instance Eq Int16X4 where
    a == b = case unpackInt16X4 a of
        (x1, x2, x3, x4) -> case unpackInt16X4 b of
            (y1, y2, y3, y4) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4

instance Ord Int16X4 where
    a `compare` b = case unpackInt16X4 a of
        (x1, x2, x3, x4) -> case unpackInt16X4 b of
            (y1, y2, y3, y4) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4

instance Show Int16X4 where
    showsPrec _ a s = case unpackInt16X4 a of
        (x1, x2, x3, x4) -> "Int16X4 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (")" ++ s))))

instance Num Int16X4 where
    (+) = plusInt16X4
    (-) = minusInt16X4
    (*) = timesInt16X4
    negate = negateInt16X4
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int16X4 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int16X4 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int16X4 where
    type Elem Int16X4 = Int16
    type ElemTuple Int16X4 = (Int16, Int16, Int16, Int16)
    nullVector         = broadcastVector 0
    vectorSize  _      = 4
    elementSize _      = 2
    broadcastVector    = broadcastInt16X4
    unsafeInsertVector = unsafeInsertInt16X4
    packVector         = packInt16X4
    unpackVector       = unpackInt16X4
    mapVector          = mapInt16X4
    zipVector          = zipInt16X4
    foldVector         = foldInt16X4

instance SIMDIntVector Int16X4 where
    quotVector = quotInt16X4
    remVector  = remInt16X4

instance Prim Int16X4 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt16X4Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt16X4Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt16X4Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt16X4OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt16X4OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt16X4OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int16X4 = V_Int16X4 (PV.Vector Int16X4)
newtype instance UV.MVector s Int16X4 = MV_Int16X4 (PMV.MVector s Int16X4)

instance Vector UV.Vector Int16X4 where
    basicUnsafeFreeze (MV_Int16X4 v) = V_Int16X4 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int16X4 v) = MV_Int16X4 <$> PV.unsafeThaw v
    basicLength (V_Int16X4 v) = PV.length v
    basicUnsafeSlice start len (V_Int16X4 v) = V_Int16X4(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int16X4 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int16X4 m) (V_Int16X4 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int16X4 where
    basicLength (MV_Int16X4 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int16X4 v) = MV_Int16X4(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int16X4 v) (MV_Int16X4 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int16X4 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int16X4 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int16X4 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int16X4 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int16X4

{-# INLINE broadcastInt16X4 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt16X4 :: Int16 -> Int16X4
broadcastInt16X4 (I16# x) = case broadcastInt16# x of
    v -> Int16X4 v v v v

{-# INLINE packInt16X4 #-}
-- | Pack the elements of a tuple into a vector.
packInt16X4 :: (Int16, Int16, Int16, Int16) -> Int16X4
packInt16X4 (I16# x1, I16# x2, I16# x3, I16# x4) = Int16X4 (packInt16# (# x1 #)) (packInt16# (# x2 #)) (packInt16# (# x3 #)) (packInt16# (# x4 #))

{-# INLINE unpackInt16X4 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt16X4 :: Int16X4 -> (Int16, Int16, Int16, Int16)
unpackInt16X4 (Int16X4 m1 m2 m3 m4) = case unpackInt16# m1 of
    (# x1 #) -> case unpackInt16# m2 of
        (# x2 #) -> case unpackInt16# m3 of
            (# x3 #) -> case unpackInt16# m4 of
                (# x4 #) -> (I16# x1, I16# x2, I16# x3, I16# x4)

{-# INLINE unsafeInsertInt16X4 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt16X4 :: Int16X4 -> Int16 -> Int -> Int16X4
unsafeInsertInt16X4 (Int16X4 m1 m2 m3 m4) (I16# y) _i@(I# ip) | _i < 1 = Int16X4 (insertInt16# m1 y (ip -# 0#)) m2 m3 m4
                                                              | _i < 2 = Int16X4 m1 (insertInt16# m2 y (ip -# 1#)) m3 m4
                                                              | _i < 3 = Int16X4 m1 m2 (insertInt16# m3 y (ip -# 2#)) m4
                                                              | otherwise = Int16X4 m1 m2 m3 (insertInt16# m4 y (ip -# 3#))

{-# INLINE[1] mapInt16X4 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt16X4 :: (Int16 -> Int16) -> Int16X4 -> Int16X4
mapInt16X4 f = mapInt16X4# (\ x -> case f (I16# x) of { I16# y -> y})

{-# RULES "mapVector abs" mapInt16X4 abs = abs #-}
{-# RULES "mapVector signum" mapInt16X4 signum = signum #-}
{-# RULES "mapVector negate" mapInt16X4 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt16X4 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt16X4 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt16X4 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt16X4 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt16X4 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt16X4 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt16X4 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt16X4 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt16X4 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt16X4# #-}
-- | Unboxed helper function.
mapInt16X4# :: (Int# -> Int#) -> Int16X4 -> Int16X4
mapInt16X4# f = \ v -> case unpackInt16X4 v of
    (I16# x1, I16# x2, I16# x3, I16# x4) -> packInt16X4 (I16# (f x1), I16# (f x2), I16# (f x3), I16# (f x4))

{-# INLINE[1] zipInt16X4 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt16X4 :: (Int16 -> Int16 -> Int16) -> Int16X4 -> Int16X4 -> Int16X4
zipInt16X4 f = \ v1 v2 -> case unpackInt16X4 v1 of
    (x1, x2, x3, x4) -> case unpackInt16X4 v2 of
        (y1, y2, y3, y4) -> packInt16X4 (f x1 y1, f x2 y2, f x3 y3, f x4 y4)

{-# RULES "zipVector +" forall a b . zipInt16X4 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt16X4 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt16X4 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt16X4 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt16X4 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt16X4 #-}
-- | Fold the elements of a vector to a single value
foldInt16X4 :: (Int16 -> Int16 -> Int16) -> Int16X4 -> Int16
foldInt16X4 f' = \ v -> case unpackInt16X4 v of
    (x1, x2, x3, x4) -> x1 `f` x2 `f` x3 `f` x4
    where f !x !y = f' x y

{-# INLINE plusInt16X4 #-}
-- | Add two vectors element-wise.
plusInt16X4 :: Int16X4 -> Int16X4 -> Int16X4
plusInt16X4 (Int16X4 m1_1 m2_1 m3_1 m4_1) (Int16X4 m1_2 m2_2 m3_2 m4_2) = Int16X4 (plusInt16# m1_1 m1_2) (plusInt16# m2_1 m2_2) (plusInt16# m3_1 m3_2) (plusInt16# m4_1 m4_2)

{-# INLINE minusInt16X4 #-}
-- | Subtract two vectors element-wise.
minusInt16X4 :: Int16X4 -> Int16X4 -> Int16X4
minusInt16X4 (Int16X4 m1_1 m2_1 m3_1 m4_1) (Int16X4 m1_2 m2_2 m3_2 m4_2) = Int16X4 (minusInt16# m1_1 m1_2) (minusInt16# m2_1 m2_2) (minusInt16# m3_1 m3_2) (minusInt16# m4_1 m4_2)

{-# INLINE timesInt16X4 #-}
-- | Multiply two vectors element-wise.
timesInt16X4 :: Int16X4 -> Int16X4 -> Int16X4
timesInt16X4 (Int16X4 m1_1 m2_1 m3_1 m4_1) (Int16X4 m1_2 m2_2 m3_2 m4_2) = Int16X4 (timesInt16# m1_1 m1_2) (timesInt16# m2_1 m2_2) (timesInt16# m3_1 m3_2) (timesInt16# m4_1 m4_2)

{-# INLINE quotInt16X4 #-}
-- | Rounds towards zero element-wise.
quotInt16X4 :: Int16X4 -> Int16X4 -> Int16X4
quotInt16X4 (Int16X4 m1_1 m2_1 m3_1 m4_1) (Int16X4 m1_2 m2_2 m3_2 m4_2) = Int16X4 (quotInt16# m1_1 m1_2) (quotInt16# m2_1 m2_2) (quotInt16# m3_1 m3_2) (quotInt16# m4_1 m4_2)

{-# INLINE remInt16X4 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt16X4 :: Int16X4 -> Int16X4 -> Int16X4
remInt16X4 (Int16X4 m1_1 m2_1 m3_1 m4_1) (Int16X4 m1_2 m2_2 m3_2 m4_2) = Int16X4 (remInt16# m1_1 m1_2) (remInt16# m2_1 m2_2) (remInt16# m3_1 m3_2) (remInt16# m4_1 m4_2)

{-# INLINE negateInt16X4 #-}
-- | Negate element-wise.
negateInt16X4 :: Int16X4 -> Int16X4
negateInt16X4 (Int16X4 m1_1 m2_1 m3_1 m4_1) = Int16X4 (negateInt16# m1_1) (negateInt16# m2_1) (negateInt16# m3_1) (negateInt16# m4_1)

{-# INLINE indexInt16X4Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt16X4Array :: ByteArray -> Int -> Int16X4
indexInt16X4Array (ByteArray a) (I# i) = Int16X4 (indexInt16Array# a ((i *# 4#) +# 0#)) (indexInt16Array# a ((i *# 4#) +# 1#)) (indexInt16Array# a ((i *# 4#) +# 2#)) (indexInt16Array# a ((i *# 4#) +# 3#))

{-# INLINE readInt16X4Array #-}
-- | Read a vector from specified index of the mutable array.
readInt16X4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int16X4
readInt16X4Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt16Array# a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> case readInt16Array# a ((i *# 4#) +# 1#) s1 of
        (# s2, m2 #) -> case readInt16Array# a ((i *# 4#) +# 2#) s2 of
            (# s3, m3 #) -> case readInt16Array# a ((i *# 4#) +# 3#) s3 of
                (# s4, m4 #) -> (# s4, Int16X4 m1 m2 m3 m4 #))

{-# INLINE writeInt16X4Array #-}
-- | Write a vector to specified index of mutable array.
writeInt16X4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int16X4 -> m ()
writeInt16X4Array (MutableByteArray a) (I# i) (Int16X4 m1 m2 m3 m4) = primitive_ (writeInt16Array# a ((i *# 4#) +# 0#) m1) >> primitive_ (writeInt16Array# a ((i *# 4#) +# 1#) m2) >> primitive_ (writeInt16Array# a ((i *# 4#) +# 2#) m3) >> primitive_ (writeInt16Array# a ((i *# 4#) +# 3#) m4)

{-# INLINE indexInt16X4OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt16X4OffAddr :: Addr -> Int -> Int16X4
indexInt16X4OffAddr (Addr a) (I# i) = Int16X4 (indexInt16OffAddr# (plusAddr# a ((i *# 8#) +# 0#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 8#) +# 2#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 8#) +# 4#)) 0#) (indexInt16OffAddr# (plusAddr# a ((i *# 8#) +# 6#)) 0#)

{-# INLINE readInt16X4OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt16X4OffAddr :: PrimMonad m => Addr -> Int -> m Int16X4
readInt16X4OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 2#) s1 of
        (# s2, m2 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 4#) s2 of
            (# s3, m3 #) -> case (\ addr i' -> readInt16OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 6#) s3 of
                (# s4, m4 #) -> (# s4, Int16X4 m1 m2 m3 m4 #))

{-# INLINE writeInt16X4OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt16X4OffAddr :: PrimMonad m => Addr -> Int -> Int16X4 -> m ()
writeInt16X4OffAddr (Addr a) (I# i) (Int16X4 m1 m2 m3 m4) = primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 8#) +# 0#)) 0# m1) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 8#) +# 2#)) 0# m2) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 8#) +# 4#)) 0# m3) >> primitive_ (writeInt16OffAddr# (plusAddr# a ((i *# 8#) +# 6#)) 0# m4)


