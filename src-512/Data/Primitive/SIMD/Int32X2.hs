{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Int32X2 (Int32X2) where

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

-- ** Int32X2
data Int32X2 = Int32X2 Int32X2# deriving Typeable

abs' :: Int32 -> Int32
abs' (I32# x) = I32# (abs# x)

{-# NOINLINE abs# #-}
abs# :: Int# -> Int#
abs# x = case abs (I32# x) of
    I32# y -> y

signum' :: Int32 -> Int32
signum' (I32# x) = I32# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Int# -> Int#
signum# x = case signum (I32# x) of
    I32# y -> y

instance Eq Int32X2 where
    a == b = case unpackInt32X2 a of
        (x1, x2) -> case unpackInt32X2 b of
            (y1, y2) -> x1 == y1 && x2 == y2

instance Ord Int32X2 where
    a `compare` b = case unpackInt32X2 a of
        (x1, x2) -> case unpackInt32X2 b of
            (y1, y2) -> x1 `compare` y1 <> x2 `compare` y2

instance Show Int32X2 where
    showsPrec _ a s = case unpackInt32X2 a of
        (x1, x2) -> "Int32X2 (" ++ shows x1 (", " ++ shows x2 (")" ++ s))

instance Num Int32X2 where
    (+) = plusInt32X2
    (-) = minusInt32X2
    (*) = timesInt32X2
    negate = negateInt32X2
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Int32X2 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Int32X2 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Int32X2 where
    type Elem Int32X2 = Int32
    type ElemTuple Int32X2 = (Int32, Int32)
    nullVector         = broadcastVector 0
    vectorSize  _      = 2
    elementSize _      = 4
    broadcastVector    = broadcastInt32X2
    unsafeInsertVector = unsafeInsertInt32X2
    packVector         = packInt32X2
    unpackVector       = unpackInt32X2
    mapVector          = mapInt32X2
    zipVector          = zipInt32X2
    foldVector         = foldInt32X2

instance SIMDIntVector Int32X2 where
    quotVector = quotInt32X2
    remVector  = remInt32X2

instance Prim Int32X2 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexInt32X2Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readInt32X2Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeInt32X2Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexInt32X2OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readInt32X2OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeInt32X2OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Int32X2 = V_Int32X2 (PV.Vector Int32X2)
newtype instance UV.MVector s Int32X2 = MV_Int32X2 (PMV.MVector s Int32X2)

instance Vector UV.Vector Int32X2 where
    basicUnsafeFreeze (MV_Int32X2 v) = V_Int32X2 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Int32X2 v) = MV_Int32X2 <$> PV.unsafeThaw v
    basicLength (V_Int32X2 v) = PV.length v
    basicUnsafeSlice start len (V_Int32X2 v) = V_Int32X2(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Int32X2 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Int32X2 m) (V_Int32X2 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Int32X2 where
    basicLength (MV_Int32X2 v) = PMV.length v
    basicUnsafeSlice start len (MV_Int32X2 v) = MV_Int32X2(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Int32X2 v) (MV_Int32X2 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Int32X2 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Int32X2 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Int32X2 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Int32X2 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Int32X2

{-# INLINE broadcastInt32X2 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastInt32X2 :: Int32 -> Int32X2
broadcastInt32X2 (I32# x) = Int32X2 (broadcastInt32X2# x)

{-# INLINE packInt32X2 #-}
-- | Pack the elements of a tuple into a vector.
packInt32X2 :: (Int32, Int32) -> Int32X2
packInt32X2 (I32# x1, I32# x2) = Int32X2 (packInt32X2# (# x1, x2 #))

{-# INLINE unpackInt32X2 #-}
-- | Unpack the elements of a vector into a tuple.
unpackInt32X2 :: Int32X2 -> (Int32, Int32)
unpackInt32X2 (Int32X2 m1) = case unpackInt32X2# m1 of
    (# x1, x2 #) -> (I32# x1, I32# x2)

{-# INLINE unsafeInsertInt32X2 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertInt32X2 :: Int32X2 -> Int32 -> Int -> Int32X2
unsafeInsertInt32X2 (Int32X2 m1) (I32# y) _i@(I# ip) = Int32X2 (insertInt32X2# m1 y (ip -# 0#))

{-# INLINE[1] mapInt32X2 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapInt32X2 :: (Int32 -> Int32) -> Int32X2 -> Int32X2
mapInt32X2 f = mapInt32X2# (\ x -> case f (I32# x) of { I32# y -> y})

{-# RULES "mapVector abs" mapInt32X2 abs = abs #-}
{-# RULES "mapVector signum" mapInt32X2 signum = signum #-}
{-# RULES "mapVector negate" mapInt32X2 negate = negate #-}
{-# RULES "mapVector const" forall x . mapInt32X2 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapInt32X2 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapInt32X2 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapInt32X2 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapInt32X2 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapInt32X2 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapInt32X2 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapInt32X2 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapInt32X2 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapInt32X2# #-}
-- | Unboxed helper function.
mapInt32X2# :: (Int# -> Int#) -> Int32X2 -> Int32X2
mapInt32X2# f = \ v -> case unpackInt32X2 v of
    (I32# x1, I32# x2) -> packInt32X2 (I32# (f x1), I32# (f x2))

{-# INLINE[1] zipInt32X2 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipInt32X2 :: (Int32 -> Int32 -> Int32) -> Int32X2 -> Int32X2 -> Int32X2
zipInt32X2 f = \ v1 v2 -> case unpackInt32X2 v1 of
    (x1, x2) -> case unpackInt32X2 v2 of
        (y1, y2) -> packInt32X2 (f x1 y1, f x2 y2)

{-# RULES "zipVector +" forall a b . zipInt32X2 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipInt32X2 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipInt32X2 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipInt32X2 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipInt32X2 rem a b = a `remVector` b #-}

{-# INLINE[1] foldInt32X2 #-}
-- | Fold the elements of a vector to a single value
foldInt32X2 :: (Int32 -> Int32 -> Int32) -> Int32X2 -> Int32
foldInt32X2 f' = \ v -> case unpackInt32X2 v of
    (x1, x2) -> x1 `f` x2
    where f !x !y = f' x y

{-# INLINE plusInt32X2 #-}
-- | Add two vectors element-wise.
plusInt32X2 :: Int32X2 -> Int32X2 -> Int32X2
plusInt32X2 (Int32X2 m1_1) (Int32X2 m1_2) = Int32X2 (plusInt32X2# m1_1 m1_2)

{-# INLINE minusInt32X2 #-}
-- | Subtract two vectors element-wise.
minusInt32X2 :: Int32X2 -> Int32X2 -> Int32X2
minusInt32X2 (Int32X2 m1_1) (Int32X2 m1_2) = Int32X2 (minusInt32X2# m1_1 m1_2)

{-# INLINE timesInt32X2 #-}
-- | Multiply two vectors element-wise.
timesInt32X2 :: Int32X2 -> Int32X2 -> Int32X2
timesInt32X2 (Int32X2 m1_1) (Int32X2 m1_2) = Int32X2 (timesInt32X2# m1_1 m1_2)

{-# INLINE quotInt32X2 #-}
-- | Rounds towards zero element-wise.
quotInt32X2 :: Int32X2 -> Int32X2 -> Int32X2
quotInt32X2 (Int32X2 m1_1) (Int32X2 m1_2) = Int32X2 (quotInt32X2# m1_1 m1_2)

{-# INLINE remInt32X2 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remInt32X2 :: Int32X2 -> Int32X2 -> Int32X2
remInt32X2 (Int32X2 m1_1) (Int32X2 m1_2) = Int32X2 (remInt32X2# m1_1 m1_2)

{-# INLINE negateInt32X2 #-}
-- | Negate element-wise.
negateInt32X2 :: Int32X2 -> Int32X2
negateInt32X2 (Int32X2 m1_1) = Int32X2 (negateInt32X2# m1_1)

{-# INLINE indexInt32X2Array #-}
-- | Read a vector from specified index of the immutable array.
indexInt32X2Array :: ByteArray -> Int -> Int32X2
indexInt32X2Array (ByteArray a) (I# i) = Int32X2 (indexInt32X2Array# a i)

{-# INLINE readInt32X2Array #-}
-- | Read a vector from specified index of the mutable array.
readInt32X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Int32X2
readInt32X2Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readInt32X2Array# a ((i *# 1#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int32X2 m1 #))

{-# INLINE writeInt32X2Array #-}
-- | Write a vector to specified index of mutable array.
writeInt32X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Int32X2 -> m ()
writeInt32X2Array (MutableByteArray a) (I# i) (Int32X2 m1) = primitive_ (writeInt32X2Array# a ((i *# 1#) +# 0#) m1)

{-# INLINE indexInt32X2OffAddr #-}
-- | Reads vector from the specified index of the address.
indexInt32X2OffAddr :: Addr -> Int -> Int32X2
indexInt32X2OffAddr (Addr a) (I# i) = Int32X2 (indexInt32X2OffAddr# (plusAddr# a (i *# 8#)) 0#)

{-# INLINE readInt32X2OffAddr #-}
-- | Reads vector from the specified index of the address.
readInt32X2OffAddr :: PrimMonad m => Addr -> Int -> m Int32X2
readInt32X2OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readInt32X2OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Int32X2 m1 #))

{-# INLINE writeInt32X2OffAddr #-}
-- | Write vector to the specified index of the address.
writeInt32X2OffAddr :: PrimMonad m => Addr -> Int -> Int32X2 -> m ()
writeInt32X2OffAddr (Addr a) (I# i) (Int32X2 m1) = primitive_ (writeInt32X2OffAddr# (plusAddr# a ((i *# 8#) +# 0#)) 0# m1)


