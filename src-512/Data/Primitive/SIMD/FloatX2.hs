{-# LANGUAGE UnboxedTuples         #-}
{-# LANGUAGE MagicHash             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE BangPatterns          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE CPP                   #-}

module Data.Primitive.SIMD.FloatX2 (FloatX2) where

-- This code was AUTOMATICALLY generated, DO NOT EDIT!

import Data.Primitive.SIMD.Class

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

-- ** FloatX2
data FloatX2 = FloatX2 FloatX2# deriving Typeable

abs' :: Float -> Float
abs' (F# x) = F# (abs# x)

{-# NOINLINE abs# #-}
abs# :: Float# -> Float#
abs# x = case abs (F# x) of
    F# y -> y

signum' :: Float -> Float
signum' (F# x) = F# (signum# x)

{-# NOINLINE signum# #-}
signum# :: Float# -> Float#
signum# x = case signum (F# x) of
    F# y -> y

instance Eq FloatX2 where
    a == b = case unpackFloatX2 a of
        (x1, x2) -> case unpackFloatX2 b of
            (y1, y2) -> x1 == y1 && x2 == y2

instance Ord FloatX2 where
    a `compare` b = case unpackFloatX2 a of
        (x1, x2) -> case unpackFloatX2 b of
            (y1, y2) -> x1 `compare` y1 <> x2 `compare` y2

instance Show FloatX2 where
    showsPrec _ a s = case unpackFloatX2 a of
        (x1, x2) -> "FloatX2 (" ++ shows x1 (", " ++ shows x2 (")" ++ s))

instance Num FloatX2 where
    (+) = plusFloatX2
    (-) = minusFloatX2
    (*) = timesFloatX2
    negate = negateFloatX2
    abs    = mapVector abs'
    signum = mapVector signum'
    fromInteger = broadcastVector . fromInteger

instance Fractional FloatX2 where
    (/)          = divideFloatX2
    recip v      = broadcastVector 1 / v
    fromRational = broadcastVector . fromRational

instance Floating FloatX2 where
    pi           = broadcastVector pi
    exp          = mapVector exp
    sqrt         = mapVector sqrt
    log          = mapVector log
    (**)         = zipVector (**)
    logBase      = zipVector (**)
    sin          = mapVector sin 
    tan          = mapVector tan
    cos          = mapVector cos 
    asin         = mapVector asin
    atan         = mapVector atan 
    acos         = mapVector acos
    sinh         = mapVector sinh 
    tanh         = mapVector tanh
    cosh         = mapVector cosh
    asinh        = mapVector asinh
    atanh        = mapVector atanh
    acosh        = mapVector acosh

instance Storable FloatX2 where
    sizeOf x     = vectorSize x * elementSize x
    alignment    = sizeOf
    peek (Ptr a) = readOffAddr (Addr a) 0
    poke (Ptr a) = writeOffAddr (Addr a) 0

instance SIMDVector FloatX2 where
    type Elem FloatX2 = Float
    type ElemTuple FloatX2 = (Float, Float)
    nullVector         = broadcastVector 0
    vectorSize  _      = 2
    elementSize _      = 4
    broadcastVector    = broadcastFloatX2
    unsafeInsertVector = unsafeInsertFloatX2
    packVector         = packFloatX2
    unpackVector       = unpackFloatX2
    mapVector          = mapFloatX2
    zipVector          = zipFloatX2
    foldVector         = foldFloatX2

instance Prim FloatX2 where
    sizeOf# a                   = let !(I# x) = sizeOf a in x
    alignment# a                = let !(I# x) = alignment a in x
    indexByteArray# ba i        = indexFloatX2Array (ByteArray ba) (I# i)
    readByteArray# mba i s      = let (ST r) = readFloatX2Array (MutableByteArray mba) (I# i) in r s
    writeByteArray# mba i v s   = let (ST r) = writeFloatX2Array (MutableByteArray mba) (I# i) v in case r s of { (# s', _ #) -> s' }
    setByteArray# mba off n v s = let (ST r) = setByteArrayGeneric (MutableByteArray mba) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }
    indexOffAddr# addr i        = indexFloatX2OffAddr (Addr addr) (I# i)
    readOffAddr# addr i s       = let (ST r) = readFloatX2OffAddr (Addr addr) (I# i) in r s
    writeOffAddr# addr i v s    = let (ST r) = writeFloatX2OffAddr (Addr addr) (I# i) v in case r s of { (# s', _ #) -> s' }
    setOffAddr# addr off n v s  = let (ST r) = setOffAddrGeneric (Addr addr) (I# off) (I# n) v in case r s of { (# s', _ #) -> s' }

newtype instance UV.Vector FloatX2 = V_FloatX2 (PV.Vector FloatX2)
newtype instance UV.MVector s FloatX2 = MV_FloatX2 (PMV.MVector s FloatX2)

instance Vector UV.Vector FloatX2 where
    basicUnsafeFreeze (MV_FloatX2 v) = V_FloatX2 <$> PV.unsafeFreeze v
    basicUnsafeThaw (V_FloatX2 v) = MV_FloatX2 <$> PV.unsafeThaw v
    basicLength (V_FloatX2 v) = PV.length v
    basicUnsafeSlice start len (V_FloatX2 v) = V_FloatX2(PV.unsafeSlice start len v)
    basicUnsafeIndexM (V_FloatX2 v) = PV.unsafeIndexM v
    basicUnsafeCopy (MV_FloatX2 m) (V_FloatX2 v) = PV.unsafeCopy m v
    elemseq _ = seq
    {-# INLINE basicUnsafeFreeze #-}
    {-# INLINE basicUnsafeThaw #-}
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicUnsafeIndexM #-}
    {-# INLINE basicUnsafeCopy #-}
    {-# INLINE elemseq #-}

instance MVector UV.MVector FloatX2 where
    basicLength (MV_FloatX2 v) = PMV.length v
    basicUnsafeSlice start len (MV_FloatX2 v) = MV_FloatX2(PMV.unsafeSlice start len v)
    basicOverlaps (MV_FloatX2 v) (MV_FloatX2 w) = PMV.overlaps v w
    basicUnsafeNew len = MV_FloatX2 <$> PMV.unsafeNew len
#if MIN_VERSION_vector(0,11,0)
    basicInitialize (MV_FloatX2 v) = basicInitialize v
#endif
    basicUnsafeRead (MV_FloatX2 v) = PMV.unsafeRead v
    basicUnsafeWrite (MV_FloatX2 v) = PMV.unsafeWrite v
    {-# INLINE basicLength #-}
    {-# INLINE basicUnsafeSlice #-}
    {-# INLINE basicOverlaps #-}
    {-# INLINE basicUnsafeNew #-}
    {-# INLINE basicUnsafeRead #-}
    {-# INLINE basicUnsafeWrite #-}

instance Unbox FloatX2

{-# INLINE broadcastFloatX2 #-}
-- | Broadcast a scalar to all elements of a vector.
broadcastFloatX2 :: Float -> FloatX2
broadcastFloatX2 (F# x) = FloatX2 (broadcastFloatX2# x)

{-# INLINE packFloatX2 #-}
-- | Pack the elements of a tuple into a vector.
packFloatX2 :: (Float, Float) -> FloatX2
packFloatX2 (F# x1, F# x2) = FloatX2 (packFloatX2# (# x1, x2 #))

{-# INLINE unpackFloatX2 #-}
-- | Unpack the elements of a vector into a tuple.
unpackFloatX2 :: FloatX2 -> (Float, Float)
unpackFloatX2 (FloatX2 m1) = case unpackFloatX2# m1 of
    (# x1, x2 #) -> (F# x1, F# x2)

{-# INLINE unsafeInsertFloatX2 #-}
-- | Insert a scalar at the given position (starting from 0) in a vector. If the index is outside of the range, the behavior is undefined.
unsafeInsertFloatX2 :: FloatX2 -> Float -> Int -> FloatX2
unsafeInsertFloatX2 (FloatX2 m1) (F# y) _i@(I# ip) = FloatX2 (insertFloatX2# m1 y (ip -# 0#))

{-# INLINE[1] mapFloatX2 #-}
-- | Apply a function to each element of a vector (unpacks and repacks the vector)
mapFloatX2 :: (Float -> Float) -> FloatX2 -> FloatX2
mapFloatX2 f = mapFloatX2# (\ x -> case f (F# x) of { F# y -> y})

{-# RULES "mapVector abs" mapFloatX2 abs = abs #-}
{-# RULES "mapVector signum" mapFloatX2 signum = signum #-}
{-# RULES "mapVector negate" mapFloatX2 negate = negate #-}
{-# RULES "mapVector const" forall x . mapFloatX2 (const x) = const (broadcastVector x) #-}
{-# RULES "mapVector (x+)" forall x v . mapFloatX2 (\ y -> x + y) v = broadcastVector x + v #-}
{-# RULES "mapVector (+x)" forall x v . mapFloatX2 (\ y -> y + x) v = v + broadcastVector x #-}
{-# RULES "mapVector (x-)" forall x v . mapFloatX2 (\ y -> x - y) v = broadcastVector x - v #-}
{-# RULES "mapVector (-x)" forall x v . mapFloatX2 (\ y -> y - x) v = v - broadcastVector x #-}
{-# RULES "mapVector (x*)" forall x v . mapFloatX2 (\ y -> x * y) v = broadcastVector x * v #-}
{-# RULES "mapVector (*x)" forall x v . mapFloatX2 (\ y -> y * x) v = v * broadcastVector x #-}
{-# RULES "mapVector (x/)" forall x v . mapFloatX2 (\ y -> x / y) v = broadcastVector x / v #-}
{-# RULES "mapVector (/x)" forall x v . mapFloatX2 (\ y -> y / x) v = v / broadcastVector x #-}

{-# INLINE[0] mapFloatX2# #-}
-- | Unboxed helper function.
mapFloatX2# :: (Float# -> Float#) -> FloatX2 -> FloatX2
mapFloatX2# f = \ v -> case unpackFloatX2 v of
    (F# x1, F# x2) -> packFloatX2 (F# (f x1), F# (f x2))

{-# INLINE[1] zipFloatX2 #-}
-- | Zip two vectors together using a combining function (unpacks and repacks the vectors)
zipFloatX2 :: (Float -> Float -> Float) -> FloatX2 -> FloatX2 -> FloatX2
zipFloatX2 f = \ v1 v2 -> case unpackFloatX2 v1 of
    (x1, x2) -> case unpackFloatX2 v2 of
        (y1, y2) -> packFloatX2 (f x1 y1, f x2 y2)

{-# RULES "zipVector +" forall a b . zipFloatX2 (+) a b = a + b #-}
{-# RULES "zipVector -" forall a b . zipFloatX2 (-) a b = a - b #-}
{-# RULES "zipVector *" forall a b . zipFloatX2 (*) a b = a * b #-}
{-# RULES "zipVector /" forall a b . zipFloatX2 (/) a b = a / b #-}

{-# INLINE[1] foldFloatX2 #-}
-- | Fold the elements of a vector to a single value
foldFloatX2 :: (Float -> Float -> Float) -> FloatX2 -> Float
foldFloatX2 f' = \ v -> case unpackFloatX2 v of
    (x1, x2) -> x1 `f` x2
    where f !x !y = f' x y

{-# INLINE plusFloatX2 #-}
-- | Add two vectors element-wise.
plusFloatX2 :: FloatX2 -> FloatX2 -> FloatX2
plusFloatX2 (FloatX2 m1_1) (FloatX2 m1_2) = FloatX2 (plusFloatX2# m1_1 m1_2)

{-# INLINE minusFloatX2 #-}
-- | Subtract two vectors element-wise.
minusFloatX2 :: FloatX2 -> FloatX2 -> FloatX2
minusFloatX2 (FloatX2 m1_1) (FloatX2 m1_2) = FloatX2 (minusFloatX2# m1_1 m1_2)

{-# INLINE timesFloatX2 #-}
-- | Multiply two vectors element-wise.
timesFloatX2 :: FloatX2 -> FloatX2 -> FloatX2
timesFloatX2 (FloatX2 m1_1) (FloatX2 m1_2) = FloatX2 (timesFloatX2# m1_1 m1_2)

{-# INLINE divideFloatX2 #-}
-- | Divide two vectors element-wise.
divideFloatX2 :: FloatX2 -> FloatX2 -> FloatX2
divideFloatX2 (FloatX2 m1_1) (FloatX2 m1_2) = FloatX2 (divideFloatX2# m1_1 m1_2)

{-# INLINE negateFloatX2 #-}
-- | Negate element-wise.
negateFloatX2 :: FloatX2 -> FloatX2
negateFloatX2 (FloatX2 m1_1) = FloatX2 (negateFloatX2# m1_1)

{-# INLINE indexFloatX2Array #-}
-- | Read a vector from specified index of the immutable array.
indexFloatX2Array :: ByteArray -> Int -> FloatX2
indexFloatX2Array (ByteArray a) (I# i) = FloatX2 (indexFloatX2Array# a i)

{-# INLINE readFloatX2Array #-}
-- | Read a vector from specified index of the mutable array.
readFloatX2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> m FloatX2
readFloatX2Array (MutableByteArray a) (I# i) = primitive (\ s0 -> case readFloatX2Array# a ((i *# 1#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, FloatX2 m1 #))

{-# INLINE writeFloatX2Array #-}
-- | Write a vector to specified index of mutable array.
writeFloatX2Array :: PrimMonad m => MutableByteArray (PrimState m) -> Int -> FloatX2 -> m ()
writeFloatX2Array (MutableByteArray a) (I# i) (FloatX2 m1) = primitive_ (writeFloatX2Array# a ((i *# 1#) +# 0#) m1)

{-# INLINE indexFloatX2OffAddr #-}
-- | Reads vector from the specified index of the address.
indexFloatX2OffAddr :: Addr -> Int -> FloatX2
indexFloatX2OffAddr (Addr a) (I# i) = FloatX2 (indexFloatX2OffAddr# (plusAddr# a (i *# 8#)) 0#)

{-# INLINE readFloatX2OffAddr #-}
-- | Reads vector from the specified index of the address.
readFloatX2OffAddr :: PrimMonad m => Addr -> Int -> m FloatX2
readFloatX2OffAddr (Addr a) (I# i) = primitive (\ s0 -> case (\ addr i' -> readFloatX2OffAddr# (plusAddr# addr i') 0#) a ((i *# 8#) +# 0#) s0 of
    (# s1, m1 #) -> (# s1, FloatX2 m1 #))

{-# INLINE writeFloatX2OffAddr #-}
-- | Write vector to the specified index of the address.
writeFloatX2OffAddr :: PrimMonad m => Addr -> Int -> FloatX2 -> m ()
writeFloatX2OffAddr (Addr a) (I# i) (FloatX2 m1) = primitive_ (writeFloatX2OffAddr# (plusAddr# a ((i *# 8#) +# 0#)) 0# m1)


