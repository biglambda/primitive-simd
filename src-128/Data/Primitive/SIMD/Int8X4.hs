{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int8X4 (Int8X4) where

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

-- ** Int8X4
data Int8X4 = Int8X4 Int8X4# deriving Typeable

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

instance Eq Int8X4 where
    a == b = case unpackInt8X4 a of
        (x1, x2, x3, x4) -> case unpackInt8X4 b of
            (y1, y2, y3, y4) -> x1 == y1 && x2 == y2 && x3 == y3 && x4 == y4

instance Ord Int8X4 where
    a `compare` b = case unpackInt8X4 a of
        (x1, x2, x3, x4) -> case unpackInt8X4 b of
            (y1, y2, y3, y4) -> x1 `compare` y1 <> x2 `compare` y2 <> x3 `compare` y3 <> x4 `compare` y4

instance Show Int8X4 where
    showsPrec _ a s = case unpackInt8X4 a of
        (x1, x2, x3, x4) -> "Int8X4 (" ++ shows x1 (", " ++ shows x2 (", " ++ shows x3 (", " ++ shows x4 (")" ++ s))))

instance Num Int8X4 where
    (+) = plusInt8X4
    (-) = minusInt8X4
    (*) = timesInt8X4
    negate = negateInt8X4
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int8X4 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int8X4 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int8X4 where
    type Elem Int8X4 = Int8
    type ElemTuple Int8X4 = (Int8, Int8, Int8, Int8)
    nullVector         = broadcastVector 0
    vectorSize  _      = 4
    elementSize _      = 1
    broadcastVector    = broadcastInt8X4
    unsafeInsertVector = unsafeInsertInt8X4
    packVector         = packInt8X4
    unpackVector       = unpackInt8X4
    mapVector          = mapInt8X4
    zipVector          = zipInt8X4
    foldVector         = foldInt8X4

instance SIMDIntVector Int8X4 where
    quotVector = quotInt8X4
    remVector  = remInt8X4

instance Prim Int8X4 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt8X4Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt8X4Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt8X4Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt8X4OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt8X4OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt8X4OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int8X4 = V_Int8X4 (PV.Vector Int8X4)
newtype instance UV.MVector s Int8X4 = MV_Int8X4 (PMV.MVector s Int8X4)

instance Vector UV.Vector Int8X4 where
    basicUnsafeFreeze (MV_Int8X4 v) = V_Int8X4 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int8X4 v) = MV_Int8X4 <$> PV.unsafeThaw v
    basicLength (V_Int8X4 v) = PV.length v
    basicUnsafeSlice start len (V_Int8X4 v) = V_Int8X4(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int8X4 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int8X4 m) (V_Int8X4 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int8X4 where
    basicLength (MV_Int8X4 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int8X4 v) = MV_Int8X4(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int8X4 v) (MV_Int8X4 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int8X4 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int8X4 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int8X4 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int8X4 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int8X4

{-# INLINE broadcastInt8X4 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt8X4 :: Int8 -> Int8X4
broadcastInt8X4 (I8# x) = Int8X4 (broadcastInt8X4# x)

{-# INLINE packInt8X4 #-}
-- | Pack the elements of a tuple into a vector.
packInt8X4 :: (Int8, Int8, Int8, Int8) -> Int8X4
packInt8X4 (I8# x1, I8# x2, I8# x3, I8# x4) = Int8X4 (packInt8X4# (# x1, x2, x3, x4 #))

{-# INLINE unpackInt8X4 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt8X4 :: Int8X4 -> (Int8, Int8, Int8, Int8)
unpackInt8X4 (Int8X4 m1) = case unpackInt8X4# m1 of
    (# x1, x2, x3, x4 #) -> (I8# x1, I8# x2, I8# x3, I8# x4)

{-# INLINE unsafeInsertInt8X4 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt8X4 :: Int8X4 -> Int8 -> Int -> Int8X4
unsafeInsertInt8X4 (Int8X4 m1) (I8# y) _i@(I# ip) = Int8X4 (insertInt8X4# m1 y (ip -# 0#))

{-# INLINE[1] mapInt8X4 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt8X4 :: (Int8 -> Int8) -> Int8X4 -> Int8X4
mapInt8X4 f = mapInt8X4# (\ x -> case f (I8# x) of { I8# y -> y})

{-# RULES "mapVector abs" mapInt8X4 abs = abs #-}
{-# RULES "mapVector signum" mapInt8X4 signum = signum #-}
{-# RULES "mapVector negate" mapInt8X4 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt8X4 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt8X4 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt8X4 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt8X4 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt8X4 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt8X4 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt8X4 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt8X4 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt8X4 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt8X4# #-}
-- | Unboxed helper function.
mapInt8X4# :: (Int# -> Int#) -> Int8X4 -> Int8X4
mapInt8X4# f = \ v -> case unpackInt8X4 v of
    (I8# x1, I8# x2, I8# x3, I8# x4) -> packInt8X4 (I8# (f x1), I8# (f x2), I8# (f x3), I8# (f x4))

{-# INLINE[1] zipInt8X4 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt8X4 :: (Int8 -> Int8 -> Int8) -> Int8X4 -> Int8X4 -> Int8X4
zipInt8X4 f = \ v1 v2 -> case unpackInt8X4 v1 of
    (x1, x2, x3, x4) -> case unpackInt8X4 v2 of
        (y1, y2, y3, y4) -> packInt8X4 (f x1 y1, f x2 y2, f x3 y3, f x4 y4)

{-# RULES "zipVector +" forall a b . zipInt8X4 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt8X4 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt8X4 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt8X4 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt8X4 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt8X4 #-}
-- | Fold the elements of a vector to a single value
foldInt8X4 :: (Int8 -> Int8 -> Int8) -> Int8X4 -> Int8
foldInt8X4 f' = \ v -> case unpackInt8X4 v of
    (x1, x2, x3, x4) -> x1 `f` x2 `f` x3 `f` x4
    where f !x !y = f' x y

{-# INLINE plusInt8X4 #-}
-- | Add two vectors element-wise.
plusInt8X4 :: Int8X4 -> Int8X4 -> Int8X4
plusInt8X4 (Int8X4 m1_1) (Int8X4 m1_2) = Int8X4 (plusInt8X4# m1_1 m1_2)

{-# INLINE minusInt8X4 #-}
-- | Subtract two vectors element-wise.
minusInt8X4 :: Int8X4 -> Int8X4 -> Int8X4
minusInt8X4 (Int8X4 m1_1) (Int8X4 m1_2) = Int8X4 (minusInt8X4# m1_1 m1_2)

{-# INLINE timesInt8X4 #-}
-- | Multiply two vectors element-wise.
timesInt8X4 :: Int8X4 -> Int8X4 -> Int8X4
timesInt8X4 (Int8X4 m1_1) (Int8X4 m1_2) = Int8X4 (timesInt8X4# m1_1 m1_2)

{-# INLINE quotInt8X4 #-}
-- | Rounds towards zero element-wise.
quotInt8X4 :: Int8X4 -> Int8X4 -> Int8X4
quotInt8X4 (Int8X4 m1_1) (Int8X4 m1_2) = Int8X4 (quotInt8X4# m1_1 m1_2)

{-# INLINE remInt8X4 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt8X4 :: Int8X4 -> Int8X4 -> Int8X4
remInt8X4 (Int8X4 m1_1) (Int8X4 m1_2) = Int8X4 (remInt8X4# m1_1 m1_2)

{-# INLINE negateInt8X4 #-}
-- | Negate element-wise.
negateInt8X4 :: Int8X4 -> Int8X4
negateInt8X4 (Int8X4 m1_1) = Int8X4 (negateInt8X4# m1_1)

{-# INLINE indexInt8X4Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt8X4Array :: ByteArray -> Int -> Int8X4
indexInt8X4Array (ByteArray a) (I# i) = Int8X4 (indexInt8X4Array# a i)

{-# INLINE readInt8X4Array #-}
-- | Read a vector from specified index of the mutable array.
readInt8X4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int8X4
readInt8X4Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt8X4Array# a ((i *# 1#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int8X4 m1 #))

{-# INLINE writeInt8X4Array #-}
-- | Write a vector to specified index of mutable array.
writeInt8X4Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int8X4 -> m ()
writeInt8X4Array (MutableByteArray a) (I# i) (Int8X4 m1) = primitive_ (writeInt8X4Array# a ((i *# 1#) +# 0#) m1)

{-# INLINE indexInt8X4OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt8X4OffAddr :: Addr -> Int -> Int8X4
indexInt8X4OffAddr (Addr a) (I# i) = Int8X4 (indexInt8X4OffAddr# (plusAddr# a (i *# 4#)) 0#)

{-# INLINE readInt8X4OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt8X4OffAddr :: PrimMonad m => Addr -> Int -> m Int8X4
readInt8X4OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt8X4OffAddr# (plusAddr# addr i') 0#) a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int8X4 m1 #))

{-# INLINE writeInt8X4OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt8X4OffAddr :: PrimMonad m => Addr -> Int -> Int8X4 -> m ()
writeInt8X4OffAddr (Addr a) (I# i) (Int8X4 m1) = primitive_ (writeInt8X4OffAddr# (plusAddr# a ((i *# 4#) +# 0#)) 0# m1)
