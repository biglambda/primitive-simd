{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.Word32X2 (Word32X2) where

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

-- ** Word32X2
data Word32X2 = Word32X2 Word32X2# deriving Typeable

abs' :: Word32 -> Word32
abs' (W32# x) = W32# (abs# x)

{-# INLINE abs# #-}
abs# :: Word# -> Word#
abs# x = case abs (W32# x) of
    W32# y -> y

signum' :: Word32 -> Word32
signum' (W32# x) = W32# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Word# -> Word#
signum# x = case signum (W32# x) of
    W32# y -> y

instance Eq Word32X2 where
    a == b = case unpackWord32X2 a of
        (x1, x2) -> case unpackWord32X2 b of
            (y1, y2) -> x1 == y1 && x2 == y2

instance Ord Word32X2 where
    a `compare` b = case unpackWord32X2 a of
        (x1, x2) -> case unpackWord32X2 b of
            (y1, y2) -> x1 `compare` y1 <> x2 `compare` y2

instance Show Word32X2 where
    showsPrec _ a s = case unpackWord32X2 a of
        (x1, x2) -> "Word32X2 (" ++ shows x1 (", " ++ shows x2 (")" ++ s))

instance Num Word32X2 where
    (+) = plusWord32X2
    (-) = minusWord32X2
    (*) = timesWord32X2
    negate = mapVector negate
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Bounded Word32X2 where
    minBound = broadcastVector minBound
    maxBound = broadcastVector maxBound

instance Storable Word32X2 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector Word32X2 where
    type Elem Word32X2 = Word32
    type ElemTuple Word32X2 = (Word32, Word32)
    nullVector         = broadcastVector 0
    vectorSize  _      = 2
    elementSize _      = 4
    broadcastVector    = broadcastWord32X2
    unsafeInsertVector = unsafeInsertWord32X2
    packVector         = packWord32X2
    unpackVector       = unpackWord32X2
    mapVector          = mapWord32X2
    zipVector          = zipWord32X2
    foldVector         = foldWord32X2

instance SIMDIntVector Word32X2 where
    quotVector = quotWord32X2
    remVector  = remWord32X2

instance Prim Word32X2 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexWord32X2Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readWord32X2Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeWord32X2Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexWord32X2OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readWord32X2OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeWord32X2OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector Word32X2 = V_Word32X2 (PV.Vector Word32X2)
newtype instance UV.MVector s Word32X2 = MV_Word32X2 (PMV.MVector s Word32X2)

instance Vector UV.Vector Word32X2 where
    basicUnsafeFreeze (MV_Word32X2 v) = V_Word32X2 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_Word32X2 v) = MV_Word32X2 <$> PV.unsafeThaw v
    basicLength (V_Word32X2 v) = PV.length v
    basicUnsafeSlice start len (V_Word32X2 v) = V_Word32X2(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_Word32X2 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_Word32X2 m) (V_Word32X2 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector Word32X2 where
    basicLength (MV_Word32X2 v) = PMV.length v
    basicUnsafeSlice start len (MV_Word32X2 v) = MV_Word32X2(PMV.unsafeSlice start len v)
    basicOverlaps (MV_Word32X2 v) (MV_Word32X2 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_Word32X2 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_Word32X2 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_Word32X2 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_Word32X2 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox Word32X2

{-# INLINE broadcastWord32X2 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastWord32X2 :: Word32 -> Word32X2
broadcastWord32X2 (W32# x) = Word32X2 (broadcastWord32X2# x)

{-# INLINE packWord32X2 #-}
-- | Pack the elements of a tuple into a vector.
packWord32X2 :: (Word32, Word32) -> Word32X2
packWord32X2 (W32# x1, W32# x2) = Word32X2 (packWord32X2# (# x1, x2 #))

{-# INLINE unpackWord32X2 #-}
-- | Unpack the elements of a vector into a tuple.
unpackWord32X2 :: Word32X2 -> (Word32, Word32)
unpackWord32X2 (Word32X2 m1) = case unpackWord32X2# m1 of
    (# x1, x2 #) -> (W32# x1, W32# x2)

{-# INLINE unsafeInsertWord32X2 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertWord32X2 :: Word32X2 -> Word32 -> Int -> Word32X2
unsafeInsertWord32X2 (Word32X2 m1) (W32# y) _i@(I# ip) = Word32X2 (insertWord32X2# m1 y (ip -# 0#))

{-# INLINE[1] mapWord32X2 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapWord32X2 :: (Word32 -> Word32) -> Word32X2 -> Word32X2
mapWord32X2 f = mapWord32X2# (\ x -> case f (W32# x) of { W32# y -> y})

{-# RULES "mapVector abs" mapWord32X2 abs = abs #-}
{-# RULES "mapVector signum" mapWord32X2 signum = signum #-}

{-# RULES "mapVector const" forall x . mapWord32X2 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapWord32X2 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapWord32X2 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapWord32X2 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapWord32X2 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapWord32X2 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapWord32X2 (\ y -> y * x) v = v * broadcastVector x #-}

{-# RULES "mapVector (`quot` x)" forall x v . mapWord32X2 (\ y -> y `quot` x) v = v `quotVector` broadcastVector x #-}
{-# RULES "mapVector (x `quot`)" forall x v . mapWord32X2 (\ y -> x `quot` y) v = broadcastVector x `quotVector` v #-}

{-# INLINE[0] mapWord32X2# #-}
-- | Unboxed helper function.
mapWord32X2# :: (Word# -> Word#) -> Word32X2 -> Word32X2
mapWord32X2# f = \ v -> case unpackWord32X2 v of
    (W32# x1, W32# x2) -> packWord32X2 (W32# (f x1), W32# (f x2))

{-# INLINE[1] zipWord32X2 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipWord32X2 :: (Word32 -> Word32 -> Word32) -> Word32X2 -> Word32X2 -> Word32X2
zipWord32X2 f = \ v1 v2 -> case unpackWord32X2 v1 of
    (x1, x2) -> case unpackWord32X2 v2 of
        (y1, y2) -> packWord32X2 (f x1 y1, f x2 y2)

{-# RULES "zipVector +" forall a b . zipWord32X2 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipWord32X2 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipWord32X2 (*) a b = a * b #-}
{-# RULES "zipVector `quotVector`" forall a b . zipWord32X2 quot a b = a `quotVector` b #-}
{-# RULES "zipVector `remVector`" forall a b . zipWord32X2 rem a b = a `remVector` b #-}

{-# INLINE[1] foldWord32X2 #-}
-- | Fold the elements of a vector to a single value
foldWord32X2 :: (Word32 -> Word32 -> Word32) -> Word32X2 -> Word32
foldWord32X2 f' = \ v -> case unpackWord32X2 v of
    (x1, x2) -> x1 `f` x2
    where f !x !y = f' x y

{-# INLINE plusWord32X2 #-}
-- | Add two vectors element-wise.
plusWord32X2 :: Word32X2 -> Word32X2 -> Word32X2
plusWord32X2 (Word32X2 m1_1) (Word32X2 m1_2) = Word32X2 (plusWord32X2# m1_1 m1_2)

{-# INLINE minusWord32X2 #-}
-- | Subtract two vectors element-wise.
minusWord32X2 :: Word32X2 -> Word32X2 -> Word32X2
minusWord32X2 (Word32X2 m1_1) (Word32X2 m1_2) = Word32X2 (minusWord32X2# m1_1 m1_2)

{-# INLINE timesWord32X2 #-}
-- | Multiply two vectors element-wise.
timesWord32X2 :: Word32X2 -> Word32X2 -> Word32X2
timesWord32X2 (Word32X2 m1_1) (Word32X2 m1_2) = Word32X2 (timesWord32X2# m1_1 m1_2)

{-# INLINE quotWord32X2 #-}
-- | Rounds towards zero element-wise.
quotWord32X2 :: Word32X2 -> Word32X2 -> Word32X2
quotWord32X2 (Word32X2 m1_1) (Word32X2 m1_2) = Word32X2 (quotWord32X2# m1_1 m1_2)

{-# INLINE remWord32X2 #-}
-- | Satisfies (quot x y) * y + (rem x y) == x.
remWord32X2 :: Word32X2 -> Word32X2 -> Word32X2
remWord32X2 (Word32X2 m1_1) (Word32X2 m1_2) = Word32X2 (remWord32X2# m1_1 m1_2)

{-# INLINE indexWord32X2Array #-}
-- | Read a vector from specified index of the immutable array.
indexWord32X2Array :: ByteArray -> Int -> Word32X2
indexWord32X2Array (ByteArray a) (I# i) = Word32X2 (indexWord32X2Array# a i)

{-# INLINE readWord32X2Array #-}
-- | Read a vector from specified index of the mutable array.
readWord32X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m Word32X2
readWord32X2Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readWord32X2Array# a ((i *# 1#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Word32X2 m1 #))

{-# INLINE writeWord32X2Array #-}
-- | Write a vector to specified index of mutable array.
writeWord32X2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> Word32X2 -> m ()
writeWord32X2Array (MutableByteArray a) (I# i) (Word32X2 m1) = primitive_ (writeWord32X2Array# a ((i *# 1#) +# 0#) m1)

{-# INLINE indexWord32X2OffAddr #-}
-- | Reads vector from the specified index of the address.
indexWord32X2OffAddr :: Addr -> Int -> Word32X2
indexWord32X2OffAddr (Addr a) (I# i) = Word32X2 (indexWord32X2OffAddr# (plusAddr# a (i *# 8#)) 0#)

{-# INLINE readWord32X2OffAddr #-}
-- | Reads vector from the specified index of the address.
readWord32X2OffAddr :: PrimMonad m => Addr -> Int -> m Word32X2
readWord32X2OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readWord32X2OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, Word32X2 m1 #))

{-# INLINE writeWord32X2OffAddr #-}
-- | Write vector to the specified index of the address.
writeWord32X2OffAddr :: PrimMonad m => Addr -> Int -> Word32X2 -> m ()
writeWord32X2OffAddr (Addr a) (I# i) (Word32X2 m1) = primitive_ (writeWord32X2OffAddr# (plusAddr# a ((i *# 8#) +# 0#)) 0# m1)
