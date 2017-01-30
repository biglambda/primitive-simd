{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word16X2 (Word16X2) where

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

-- ** Word16X2
data Word16X2 = Word16X2 Word16X2# deriving Typeable

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

instance Eq Word16X2 where
    a == b = case unpackWord16X2 a of
        (x1, x2) -> case unpackWord16X2 b of
            (y1, y2) -> x1 == y1 && x2 == y2

instance Ord Word16X2 where
    a `compare` b = case unpackWord16X2 a of
        (x1, x2) -> case unpackWord16X2 b of
            (y1, y2) -> x1 `compare` y1 <> x2 `compare` y2

instance Show Word16X2 where
    showsPrec _ a s = case unpackWord16X2 a of
        (x1, x2) -> "Word16X2 (" ++ shows x1 (", " ++ shows x2 (")" ++ s))

instance Num Word16X2 where
    (+) = plusWord16X2
    (-) = minusWord16X2
    (*) = timesWord16X2
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word16X2 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word16X2 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word16X2 where
    type Elem Word16X2 = Word16
    type ElemTuple Word16X2 = (Word16, Word16)
    nullVector         = broadcastVector 0
    vectorSize  _      = 2
    elementSize _      = 2
    broadcastVector    = broadcastWord16X2
    unsafeInsertVector = unsafeInsertWord16X2
    packVector         = packWord16X2
    unpackVector       = unpackWord16X2
    mapVector          = mapWord16X2
    zipVector          = zipWord16X2
    foldVector         = foldWord16X2

instance SIMDIntVector Word16X2 where
    quotVector = quotWord16X2
    remVector  = remWord16X2

instance Prim Word16X2 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord16X2Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord16X2Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord16X2Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord16X2OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord16X2OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord16X2OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word16X2 = V_Word16X2 (PV.Vector Word16X2)
newtype instance UV.MVector s Word16X2 = MV_Word16X2 (PMV.MVector s Word16X2)

instance Vector UV.Vector Word16X2 where
    basicUnsafeFreeze (MV_Word16X2 v) = V_Word16X2 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word16X2 v) = MV_Word16X2 <$> PV.unsafeThaw v
    basicLength (V_Word16X2 v) = PV.length v
    basicUnsafeSlice start len (V_Word16X2 v) = V_Word16X2(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word16X2 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word16X2 m) (V_Word16X2 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word16X2 where
    basicLength (MV_Word16X2 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word16X2 v) = MV_Word16X2(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word16X2 v) (MV_Word16X2 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word16X2 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word16X2 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word16X2 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word16X2 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word16X2

{-# INLINE broadcastWord16X2 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord16X2 :: Word16 -> Word16X2
broadcastWord16X2 (W16# x) = Word16X2 (broadcastWord16X2# x)

{-# INLINE packWord16X2 #-}
-- | Pack the elements of a tuple into a vector.
packWord16X2 :: (Word16, Word16) -> Word16X2
packWord16X2 (W16# x1, W16# x2) = Word16X2 (packWord16X2# (# x1, x2 #))

{-# INLINE unpackWord16X2 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord16X2 :: Word16X2 -> (Word16, Word16)
unpackWord16X2 (Word16X2 m1) = case unpackWord16X2# m1 of
    (# x1, x2 #) -> (W16# x1, W16# x2)

{-# INLINE unsafeInsertWord16X2 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord16X2 :: Word16X2 -> Word16 -> Int -> Word16X2
unsafeInsertWord16X2 (Word16X2 m1) (W16# y) _i@(I# ip) = Word16X2 (insertWord16X2# m1 y (ip -# 0#))

{-# INLINE[1] mapWord16X2 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord16X2 :: (Word16 -> Word16) -> Word16X2 -> Word16X2
mapWord16X2 f = mapWord16X2# (\ x -> case f (W16# x) of { W16# y -> y})

{-# RULES "mapVector abs" mapWord16X2 abs = abs #-}
{-# RULES "mapVector signum" mapWord16X2 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord16X2 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord16X2 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord16X2 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord16X2 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord16X2 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord16X2 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord16X2 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord16X2 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord16X2 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord16X2# #-}
-- | Unboxed helper function.
mapWord16X2# :: (Word# -> Word#) -> Word16X2 -> Word16X2
mapWord16X2# f = \ v -> case unpackWord16X2 v of
    (W16# x1, W16# x2) -> packWord16X2 (W16# (f x1), W16# (f x2))

{-# INLINE[1] zipWord16X2 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord16X2 :: (Word16 -> Word16 -> Word16) -> Word16X2 -> Word16X2 -> Word16X2
zipWord16X2 f = \ v1 v2 -> case unpackWord16X2 v1 of
    (x1, x2) -> case unpackWord16X2 v2 of
        (y1, y2) -> packWord16X2 (f x1 y1, f x2 y2)

{-# RULES "zipVector +" forall a b . zipWord16X2 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord16X2 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord16X2 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord16X2 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord16X2 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord16X2 #-}
-- | Fold the elements of a vector to a single value
foldWord16X2 :: (Word16 -> Word16 -> Word16) -> Word16X2 -> Word16
foldWord16X2 f' = \ v -> case unpackWord16X2 v of
    (x1, x2) -> x1 `f` x2
    where f !x !y = f' x y

{-# INLINE plusWord16X2 #-}
-- | Add two vectors element-wise.
plusWord16X2 :: Word16X2 -> Word16X2 -> Word16X2
plusWord16X2 (Word16X2 m1_1) (Word16X2 m1_2) = Word16X2 (plusWord16X2# m1_1 m1_2)

{-# INLINE minusWord16X2 #-}
-- | Subtract two vectors element-wise.
minusWord16X2 :: Word16X2 -> Word16X2 -> Word16X2
minusWord16X2 (Word16X2 m1_1) (Word16X2 m1_2) = Word16X2 (minusWord16X2# m1_1 m1_2)

{-# INLINE timesWord16X2 #-}
-- | Multiply two vectors element-wise.
timesWord16X2 :: Word16X2 -> Word16X2 -> Word16X2
timesWord16X2 (Word16X2 m1_1) (Word16X2 m1_2) = Word16X2 (timesWord16X2# m1_1 m1_2)

{-# INLINE quotWord16X2 #-}
-- | Rounds towards zero element-wise.
quotWord16X2 :: Word16X2 -> Word16X2 -> Word16X2
quotWord16X2 (Word16X2 m1_1) (Word16X2 m1_2) = Word16X2 (quotWord16X2# m1_1 m1_2)

{-# INLINE remWord16X2 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord16X2 :: Word16X2 -> Word16X2 -> Word16X2
remWord16X2 (Word16X2 m1_1) (Word16X2 m1_2) = Word16X2 (remWord16X2# m1_1 m1_2)

{-# INLINE indexWord16X2Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord16X2Array :: ByteArray -> Int -> Word16X2
indexWord16X2Array (ByteArray a) (I# i) = Word16X2 (indexWord16X2Array# a i)

{-# INLINE readWord16X2Array #-}
-- | Read a vector from specified index of the mutable array.
readWord16X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word16X2
readWord16X2Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord16X2Array# a ((i *# 1#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Word16X2 m1 #))

{-# INLINE writeWord16X2Array #-}
-- | Write a vector to specified index of mutable array.
writeWord16X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word16X2 -> m ()
writeWord16X2Array (MutableByteArray a) (I# i) (Word16X2 m1) = primitive_ (writeWord16X2Array# a ((i *# 1#) +# 0#) m1)

{-# INLINE indexWord16X2OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord16X2OffAddr :: Addr -> Int -> Word16X2
indexWord16X2OffAddr (Addr a) (I# i) = Word16X2 (indexWord16X2OffAddr# (plusAddr# a (i *# 4#)) 0#)

{-# INLINE readWord16X2OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord16X2OffAddr :: PrimMonad m => Addr -> Int -> m Word16X2
readWord16X2OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord16X2OffAddr# (plusAddr# addr i') 0#) a ((i *# 4#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Word16X2 m1 #))

{-# INLINE writeWord16X2OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord16X2OffAddr :: PrimMonad m => Addr -> Int -> Word16X2 -> m ()
writeWord16X2OffAddr (Addr a) (I# i) (Word16X2 m1) = primitive_ (writeWord16X2OffAddr# (plusAddr# a ((i *# 4#) +# 0#)) 0# m1)


