ě"
é#¸#
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
Ž
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
:
OnesLike
x"T
y"T"
Ttype:
2	

M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Ľ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeíout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨

StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéčelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint˙˙˙˙˙˙˙˙˙
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.14.02v2.14.0-rc1-21-g4dacf3f368e8ŕő 

lstm_3/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namelstm_3/lstm_cell/bias
|
)lstm_3/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell/bias*
_output_shapes	
:*
dtype0
 
!lstm_3/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!lstm_3/lstm_cell/recurrent_kernel

5lstm_3/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp!lstm_3/lstm_cell/recurrent_kernel* 
_output_shapes
:
*
dtype0

lstm_3/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_namelstm_3/lstm_cell/kernel

+lstm_3/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm_3/lstm_cell/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:×8*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:×8*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
×8*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
×8*
dtype0

attention_layer/V_aVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameattention_layer/V_a
|
'attention_layer/V_a/Read/ReadVariableOpReadVariableOpattention_layer/V_a*
_output_shapes
:	*
dtype0

attention_layer/U_aVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameattention_layer/U_a
}
'attention_layer/U_a/Read/ReadVariableOpReadVariableOpattention_layer/U_a* 
_output_shapes
:
*
dtype0

attention_layer/W_aVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameattention_layer/W_a
}
'attention_layer/W_a/Read/ReadVariableOpReadVariableOpattention_layer/W_a* 
_output_shapes
:
*
dtype0

embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
×8*'
shared_nameembedding_1/embeddings

*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings* 
_output_shapes
:
×8*
dtype0

serving_default_input_2Placeholder*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0*%
shape:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
|
serving_default_input_6Placeholder*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
|
serving_default_input_7Placeholder*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

serving_default_input_8Placeholder*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*"
shape:˙˙˙˙˙˙˙˙˙

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2serving_default_input_6serving_default_input_7serving_default_input_8embedding_1/embeddingslstm_3/lstm_cell/kernellstm_3/lstm_cell/bias!lstm_3/lstm_cell/recurrent_kernelattention_layer/W_aattention_layer/U_aattention_layer/V_adense_1/kerneldense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_53794

NoOpNoOp
Ë+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*+
valueü*Bů* Bň*

layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
 
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings*
* 
* 
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
 cell
!
state_spec*
* 
Ť
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(W_a
)U_a
*V_a*

+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses* 
Ś
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias*
C
0
91
:2
;3
(4
)5
*6
77
88*
C
0
91
:2
;3
(4
)5
*6
77
88*
* 
°
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Atrace_0
Btrace_1* 

Ctrace_0
Dtrace_1* 
* 

Eserving_default* 

0*

0*
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ktrace_0* 

Ltrace_0* 
jd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

90
:1
;2*

90
:1
;2*
* 


Mstates
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Strace_0
Ttrace_1
Utrace_2
Vtrace_3* 
6
Wtrace_0
Xtrace_1
Ytrace_2
Ztrace_3* 
* 
ă
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_random_generator
b
state_size

9kernel
:recurrent_kernel
;bias*
* 

(0
)1
*2*

(0
)1
*2*
* 

cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

htrace_0* 

itrace_0* 
`Z
VARIABLE_VALUEattention_layer/W_a3layer_with_weights-2/W_a/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEattention_layer/U_a3layer_with_weights-2/U_a/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEattention_layer/V_a3layer_with_weights-2/V_a/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

otrace_0* 

ptrace_0* 

70
81*

70
81*
* 

qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

vtrace_0* 

wtrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_3/lstm_cell/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUE!lstm_3/lstm_cell/recurrent_kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUElstm_3/lstm_cell/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
* 
C
0
1
2
3
4
5
6
7
	8*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

 0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

90
:1
;2*

90
:1
;2*
* 

xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

}trace_0
~trace_1* 

trace_0
trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
î
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameembedding_1/embeddingsattention_layer/W_aattention_layer/U_aattention_layer/V_adense_1/kerneldense_1/biaslstm_3/lstm_cell/kernel!lstm_3/lstm_cell/recurrent_kernellstm_3/lstm_cell/biasConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_55726
é
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding_1/embeddingsattention_layer/W_aattention_layer/U_aattention_layer/V_adense_1/kerneldense_1/biaslstm_3/lstm_cell/kernel!lstm_3/lstm_cell/recurrent_kernellstm_3/lstm_cell/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_55762ÓĽ 
Ë~
Š
D__inference_lstm_cell_layer_call_and_return_conditional_losses_55567

inputs
states_0
states_11
split_readvariableop_resource:
.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpP
	ones_likeOnesLikeinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?l
dropout/MulMulones_like:y:0dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
dropout/ShapeShapeones_like:y:0*
T0*
_output_shapes
::íĎ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?p
dropout_1/MulMulones_like:y:0dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
dropout_1/ShapeShapeones_like:y:0*
T0*
_output_shapes
::íĎ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>­
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_1/SelectV2SelectV2dropout_1/GreaterEqual:z:0dropout_1/Mul:z:0dropout_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?p
dropout_2/MulMulones_like:y:0dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
dropout_2/ShapeShapeones_like:y:0*
T0*
_output_shapes
::íĎ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>­
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_2/SelectV2SelectV2dropout_2/GreaterEqual:z:0dropout_2/Mul:z:0dropout_2/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?p
dropout_3/MulMulones_like:y:0dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
dropout_3/ShapeShapeones_like:y:0*
T0*
_output_shapes
::íĎ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>­
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_3/SelectV2SelectV2dropout_3/GreaterEqual:z:0dropout_3/Mul:z:0dropout_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
ones_like_1OnesLikestates_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?r
dropout_4/MulMulones_like_1:y:0dropout_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
dropout_4/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::íĎ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_4/SelectV2SelectV2dropout_4/GreaterEqual:z:0dropout_4/Mul:z:0dropout_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?r
dropout_5/MulMulones_like_1:y:0dropout_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
dropout_5/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::íĎ
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_5/SelectV2SelectV2dropout_5/GreaterEqual:z:0dropout_5/Mul:z:0dropout_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?r
dropout_6/MulMulones_like_1:y:0dropout_6/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
dropout_6/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::íĎ
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_6/SelectV2SelectV2dropout_6/GreaterEqual:z:0dropout_6/Mul:z:0dropout_6/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?r
dropout_7/MulMulones_like_1:y:0dropout_7/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
dropout_7/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::íĎ
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_7/SelectV2SelectV2dropout_7/GreaterEqual:z:0dropout_7/Mul:z:0dropout_7/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
mulMulinputsdropout/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
mul_1Mulinputsdropout_1/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
mul_2Mulinputsdropout_2/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
mul_3Mulinputsdropout_3/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype0Ś
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
mul_4Mulstates_0dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
mul_5Mulstates_0dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
mul_6Mulstates_0dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
mul_7Mulstates_0dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
áy

J__inference_attention_layer_layer_call_and_return_conditional_losses_53316

inputs
inputs_13
shape_2_readvariableop_resource:
4
 matmul_1_readvariableop_resource:
2
shape_4_readvariableop_resource:	
identity

identity_1˘MatMul_1/ReadVariableOp˘transpose_1/ReadVariableOp˘transpose_2/ReadVariableOp˘whileW
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :e
SumSuminputsSum/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :i
Sum_1Suminputs Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_1transpose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::íĎ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskK
Shape_1Shapeinputs*
T0*
_output_shapes
::íĎS
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource* 
_output_shapes
:
*
dtype0X
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"      S
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
transpose_1/ReadVariableOpReadVariableOpshape_2_readvariableop_resource* 
_output_shapes
:
*
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0* 
_output_shapes
:
`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙j
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
i
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :T
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:x
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
MatMul_1MatMulstrided_slice_1:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :|

ExpandDims
ExpandDimsMatMul_1:product:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙m
addAddV2Reshape_2:output:0ExpandDims:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙M
TanhTanhadd:z:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙M
Shape_3ShapeTanh:y:0*
T0*
_output_shapes
::íĎU
	unstack_2UnpackShape_3:output:0*
T0*
_output_shapes
: : : *	
numw
Shape_4/ReadVariableOpReadVariableOpshape_4_readvariableop_resource*
_output_shapes
:	*
dtype0X
Shape_4Const*
_output_shapes
:*
dtype0*
valueB"      S
	unstack_3UnpackShape_4:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   k
	Reshape_3ReshapeTanh:y:0Reshape_3/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
transpose_2/ReadVariableOpReadVariableOpshape_4_readvariableop_resource*
_output_shapes
:	*
dtype0a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes
:	`
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙i
	Reshape_4Reshapetranspose_2:y:0Reshape_4/shape:output:0*
T0*
_output_shapes
:	l
MatMul_2MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙T
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value
B :S
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:y
	Reshape_5ReshapeMatMul_2:product:0Reshape_5/shape:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙y
SqueezeSqueezeReshape_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims

˙˙˙˙˙˙˙˙˙W
SoftmaxSoftmaxSqueeze:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ż
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0Sum_1:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0inputsshape_2_readvariableop_resource matmul_1_readvariableop_resourceshape_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙: : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_53126*
condR
while_cond_53125*R
output_shapesA
?: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙: : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_3	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_3/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙e
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_4	Transposetranspose_3:y:0transpose_4/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙T
Shape_5Shapetranspose_4:y:0*
T0*
_output_shapes
::íĎ_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_3StridedSliceShape_5:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
TensorArrayV2_3/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙¸
TensorArrayV2_3TensorListReserve&TensorArrayV2_3/element_shape:output:0strided_slice_3:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  ć
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_4:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
strided_slice_4StridedSlicetranspose_4:y:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask[
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
ExpandDims_1
ExpandDimsstrided_slice_4:output:0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙a
mulMulinputsExpandDims_1:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :j
Sum_2Summul:z:0 Sum_2/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_4/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_4TensorListReserve&TensorArrayV2_4/element_shape:output:0strided_slice_3:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇH
time_1Const*
_output_shapes
: *
dtype0*
value	B : e
while_1/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙V
while_1/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ţ
while_1StatelessWhilewhile_1/loop_counter:output:0#while_1/maximum_iterations:output:0time_1:output:0TensorArrayV2_4:handle:0Sum:output:0strided_slice_3:output:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0inputs*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*M
_output_shapes;
9: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *
bodyR
while_1_body_53258*
condR
while_1_cond_53257*L
output_shapes;
9: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙*
parallel_iterations 
2TensorArrayV2Stack_1/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ň
$TensorArrayV2Stack_1/TensorListStackTensorListStackwhile_1:output:3;TensorArrayV2Stack_1/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_5StridedSlice-TensorArrayV2Stack_1/TensorListStack:tensor:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_5/permConst*
_output_shapes
:*
dtype0*!
valueB"          ˘
transpose_5	Transpose-TensorArrayV2Stack_1/TensorListStack:tensor:0transpose_5/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙l
IdentityIdentitytranspose_5:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙n

Identity_1Identitytranspose_3:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙~
NoOpNoOp^MatMul_1/ReadVariableOp^transpose_1/ReadVariableOp^transpose_2/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_2/ReadVariableOptranspose_2/ReadVariableOp2
whilewhile:U Q
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:]Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ŕ
×
while_1_cond_55279 
while_1_while_1_loop_counter&
"while_1_while_1_maximum_iterations
while_1_placeholder
while_1_placeholder_1
while_1_placeholder_2 
while_1_less_strided_slice_37
3while_1_while_1_cond_55279___redundant_placeholder07
3while_1_while_1_cond_55279___redundant_placeholder1
while_1_identity
h
while_1/LessLesswhile_1_placeholderwhile_1_less_strided_slice_3*
T0*
_output_shapes
: O
while_1/IdentityIdentitywhile_1/Less:z:0*
T0
*
_output_shapes
: "-
while_1_identitywhile_1/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :˙˙˙˙˙˙˙˙˙: :::L H

_output_shapes
: 
.
_user_specified_namewhile_1/loop_counter:RN

_output_shapes
: 
4
_user_specified_namewhile_1/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_3:

_output_shapes
::

_output_shapes
:
ł˝
˙
while_body_52843
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
˘while/lstm_cell/ReadVariableOp˘ while/lstm_cell/ReadVariableOp_1˘ while/lstm_cell/ReadVariableOp_2˘ while/lstm_cell/ReadVariableOp_3˘$while/lstm_cell/split/ReadVariableOp˘&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0
while/lstm_cell/ones_likeOnesLike0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
while/lstm_cell/dropout/MulMulwhile/lstm_cell/ones_like:y:0&while/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
while/lstm_cell/dropout/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎ­
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>×
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ô
 while/lstm_cell/dropout/SelectV2SelectV2(while/lstm_cell/dropout/GreaterEqual:z:0while/lstm_cell/dropout/Mul:z:0(while/lstm_cell/dropout/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ? 
while/lstm_cell/dropout_1/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/lstm_cell/dropout_1/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ý
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_1/SelectV2SelectV2*while/lstm_cell/dropout_1/GreaterEqual:z:0!while/lstm_cell/dropout_1/Mul:z:0*while/lstm_cell/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ? 
while/lstm_cell/dropout_2/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/lstm_cell/dropout_2/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ý
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_2/SelectV2SelectV2*while/lstm_cell/dropout_2/GreaterEqual:z:0!while/lstm_cell/dropout_2/Mul:z:0*while/lstm_cell/dropout_2/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ? 
while/lstm_cell/dropout_3/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/lstm_cell/dropout_3/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ý
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_3/SelectV2SelectV2*while/lstm_cell/dropout_3/GreaterEqual:z:0!while/lstm_cell/dropout_3/Mul:z:0*while/lstm_cell/dropout_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
while/lstm_cell/ones_like_1OnesLikewhile_placeholder_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
while/lstm_cell/dropout_4/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
while/lstm_cell/dropout_4/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_4/SelectV2SelectV2*while/lstm_cell/dropout_4/GreaterEqual:z:0!while/lstm_cell/dropout_4/Mul:z:0*while/lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
while/lstm_cell/dropout_5/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
while/lstm_cell/dropout_5/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_5/SelectV2SelectV2*while/lstm_cell/dropout_5/GreaterEqual:z:0!while/lstm_cell/dropout_5/Mul:z:0*while/lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
while/lstm_cell/dropout_6/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
while/lstm_cell/dropout_6/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_6/SelectV2SelectV2*while/lstm_cell/dropout_6/GreaterEqual:z:0!while/lstm_cell/dropout_6/Mul:z:0*while/lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
while/lstm_cell/dropout_7/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
while/lstm_cell/dropout_7/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_7/SelectV2SelectV2*while/lstm_cell/dropout_7/GreaterEqual:z:0!while/lstm_cell/dropout_7/Mul:z:0*while/lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/lstm_cell/dropout/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ž
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_1/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ž
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_2/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ž
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_3/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ö
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Č
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_4Mulwhile_placeholder_2+while/lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_5Mulwhile_placeholder_2+while/lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_6Mulwhile_placeholder_2+while/lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_7Mulwhile_placeholder_2+while/lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ˝
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
Ă
Ž
#__inference_signature_wrapper_53794
input_2
input_6
input_7
input_8
unknown:
×8
	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:
×8
	unknown_7:	×8
identity

identity_1

identity_2˘StatefulPartitionedCallę
StatefulPartitionedCallStatefulPartitionedCallinput_2input_8input_6input_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_52154}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_6:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_7:VR
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8:%!

_user_specified_name53770:%!

_user_specified_name53772:%!

_user_specified_name53774:%!

_user_specified_name53776:%!

_user_specified_name53778:%	!

_user_specified_name53780:%
!

_user_specified_name53782:%!

_user_specified_name53784:%!

_user_specified_name53786
ăo
˙
while_body_54924
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
˘while/lstm_cell/ReadVariableOp˘ while/lstm_cell/ReadVariableOp_1˘ while/lstm_cell/ReadVariableOp_2˘ while/lstm_cell/ReadVariableOp_3˘$while/lstm_cell/split/ReadVariableOp˘&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0
while/lstm_cell/ones_likeOnesLike0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
while/lstm_cell/ones_like_1OnesLikewhile_placeholder_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ö
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Č
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_4Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_5Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_6Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_7Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ˝
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
ˇ	
Ř
while_cond_55147
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_55147___redundant_placeholder03
/while_while_cond_55147___redundant_placeholder13
/while_while_cond_55147___redundant_placeholder23
/while_while_cond_55147___redundant_placeholder33
/while_while_cond_55147___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :˙˙˙˙˙˙˙˙˙: ::::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
::

_output_shapes
:
é:

A__inference_lstm_3_layer_call_and_return_conditional_losses_52604

inputs#
lstm_cell_52520:

lstm_cell_52522:	#
lstm_cell_52524:

identity

identity_1

identity_2˘!lstm_cell/StatefulPartitionedCall˘whileI
ShapeShapeinputs*
T0*
_output_shapes
::íĎ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::íĎ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskč
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_52520lstm_cell_52522lstm_cell_52524*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_52519n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_52520lstm_cell_52522lstm_cell_52524*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_52533*
condR
while_cond_52532*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:%!

_user_specified_name52520:%!

_user_specified_name52522:%!

_user_specified_name52524
ť	
ź
while_cond_53473
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_53473___redundant_placeholder03
/while_while_cond_53473___redundant_placeholder13
/while_while_cond_53473___redundant_placeholder23
/while_while_cond_53473___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
:
˝˝
	
while_body_54043
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
˘while/lstm_cell/ReadVariableOp˘ while/lstm_cell/ReadVariableOp_1˘ while/lstm_cell/ReadVariableOp_2˘ while/lstm_cell/ReadVariableOp_3˘$while/lstm_cell/split/ReadVariableOp˘&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0
while/lstm_cell/ones_likeOnesLike0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
while/lstm_cell/dropout/MulMulwhile/lstm_cell/ones_like:y:0&while/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
while/lstm_cell/dropout/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎ­
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>×
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ô
 while/lstm_cell/dropout/SelectV2SelectV2(while/lstm_cell/dropout/GreaterEqual:z:0while/lstm_cell/dropout/Mul:z:0(while/lstm_cell/dropout/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ? 
while/lstm_cell/dropout_1/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/lstm_cell/dropout_1/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ý
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_1/SelectV2SelectV2*while/lstm_cell/dropout_1/GreaterEqual:z:0!while/lstm_cell/dropout_1/Mul:z:0*while/lstm_cell/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ? 
while/lstm_cell/dropout_2/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/lstm_cell/dropout_2/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ý
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_2/SelectV2SelectV2*while/lstm_cell/dropout_2/GreaterEqual:z:0!while/lstm_cell/dropout_2/Mul:z:0*while/lstm_cell/dropout_2/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ? 
while/lstm_cell/dropout_3/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/lstm_cell/dropout_3/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ý
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_3/SelectV2SelectV2*while/lstm_cell/dropout_3/GreaterEqual:z:0!while/lstm_cell/dropout_3/Mul:z:0*while/lstm_cell/dropout_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
while/lstm_cell/ones_like_1OnesLikewhile_placeholder_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
while/lstm_cell/dropout_4/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
while/lstm_cell/dropout_4/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_4/SelectV2SelectV2*while/lstm_cell/dropout_4/GreaterEqual:z:0!while/lstm_cell/dropout_4/Mul:z:0*while/lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
while/lstm_cell/dropout_5/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
while/lstm_cell/dropout_5/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_5/SelectV2SelectV2*while/lstm_cell/dropout_5/GreaterEqual:z:0!while/lstm_cell/dropout_5/Mul:z:0*while/lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
while/lstm_cell/dropout_6/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
while/lstm_cell/dropout_6/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_6/SelectV2SelectV2*while/lstm_cell/dropout_6/GreaterEqual:z:0!while/lstm_cell/dropout_6/Mul:z:0*while/lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
while/lstm_cell/dropout_7/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
while/lstm_cell/dropout_7/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_7/SelectV2SelectV2*while/lstm_cell/dropout_7/GreaterEqual:z:0!while/lstm_cell/dropout_7/Mul:z:0*while/lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/lstm_cell/dropout/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ž
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_1/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ž
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_2/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ž
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_3/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ö
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Č
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_4Mulwhile_placeholder_2+while/lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_5Mulwhile_placeholder_2+while/lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_6Mulwhile_placeholder_2+while/lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_7Mulwhile_placeholder_2+while/lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ˝
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource

˙
&__inference_lstm_3_layer_call_fn_53874

inputs
initial_state_0
initial_state_1
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1

identity_2˘StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsinitial_state_0initial_state_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_53606}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:YU
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinitial_state_0:YU
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinitial_state_1:%!

_user_specified_name53862:%!

_user_specified_name53864:%!

_user_specified_name53866
ó
R
&__inference_concat_layer_call_fn_55344
inputs_0
inputs_1
identityĘ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_53331n
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0:_[
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1
í

model_5_lstm_3_while_body_51731:
6model_5_lstm_3_while_model_5_lstm_3_while_loop_counter@
<model_5_lstm_3_while_model_5_lstm_3_while_maximum_iterations$
 model_5_lstm_3_while_placeholder&
"model_5_lstm_3_while_placeholder_1&
"model_5_lstm_3_while_placeholder_2&
"model_5_lstm_3_while_placeholder_37
3model_5_lstm_3_while_model_5_lstm_3_strided_slice_0u
qmodel_5_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_5_lstm_3_tensorarrayunstack_tensorlistfromtensor_0R
>model_5_lstm_3_while_lstm_cell_split_readvariableop_resource_0:
O
@model_5_lstm_3_while_lstm_cell_split_1_readvariableop_resource_0:	L
8model_5_lstm_3_while_lstm_cell_readvariableop_resource_0:
!
model_5_lstm_3_while_identity#
model_5_lstm_3_while_identity_1#
model_5_lstm_3_while_identity_2#
model_5_lstm_3_while_identity_3#
model_5_lstm_3_while_identity_4#
model_5_lstm_3_while_identity_55
1model_5_lstm_3_while_model_5_lstm_3_strided_slices
omodel_5_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_5_lstm_3_tensorarrayunstack_tensorlistfromtensorP
<model_5_lstm_3_while_lstm_cell_split_readvariableop_resource:
M
>model_5_lstm_3_while_lstm_cell_split_1_readvariableop_resource:	J
6model_5_lstm_3_while_lstm_cell_readvariableop_resource:
˘-model_5/lstm_3/while/lstm_cell/ReadVariableOp˘/model_5/lstm_3/while/lstm_cell/ReadVariableOp_1˘/model_5/lstm_3/while/lstm_cell/ReadVariableOp_2˘/model_5/lstm_3/while/lstm_cell/ReadVariableOp_3˘3model_5/lstm_3/while/lstm_cell/split/ReadVariableOp˘5model_5/lstm_3/while/lstm_cell/split_1/ReadVariableOp
Fmodel_5/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ň
8model_5/lstm_3/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqmodel_5_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_5_lstm_3_tensorarrayunstack_tensorlistfromtensor_0 model_5_lstm_3_while_placeholderOmodel_5/lstm_3/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0¨
(model_5/lstm_3/while/lstm_cell/ones_likeOnesLike?model_5/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
*model_5/lstm_3/while/lstm_cell/ones_like_1OnesLike"model_5_lstm_3_while_placeholder_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ë
"model_5/lstm_3/while/lstm_cell/mulMul?model_5/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0,model_5/lstm_3/while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
$model_5/lstm_3/while/lstm_cell/mul_1Mul?model_5/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0,model_5/lstm_3/while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
$model_5/lstm_3/while/lstm_cell/mul_2Mul?model_5/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0,model_5/lstm_3/while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Í
$model_5/lstm_3/while/lstm_cell/mul_3Mul?model_5/lstm_3/while/TensorArrayV2Read/TensorListGetItem:item:0,model_5/lstm_3/while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
.model_5/lstm_3/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :´
3model_5/lstm_3/while/lstm_cell/split/ReadVariableOpReadVariableOp>model_5_lstm_3_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
$model_5/lstm_3/while/lstm_cell/splitSplit7model_5/lstm_3/while/lstm_cell/split/split_dim:output:0;model_5/lstm_3/while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitš
%model_5/lstm_3/while/lstm_cell/MatMulMatMul&model_5/lstm_3/while/lstm_cell/mul:z:0-model_5/lstm_3/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˝
'model_5/lstm_3/while/lstm_cell/MatMul_1MatMul(model_5/lstm_3/while/lstm_cell/mul_1:z:0-model_5/lstm_3/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˝
'model_5/lstm_3/while/lstm_cell/MatMul_2MatMul(model_5/lstm_3/while/lstm_cell/mul_2:z:0-model_5/lstm_3/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˝
'model_5/lstm_3/while/lstm_cell/MatMul_3MatMul(model_5/lstm_3/while/lstm_cell/mul_3:z:0-model_5/lstm_3/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
0model_5/lstm_3/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ł
5model_5/lstm_3/while/lstm_cell/split_1/ReadVariableOpReadVariableOp@model_5_lstm_3_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0ő
&model_5/lstm_3/while/lstm_cell/split_1Split9model_5/lstm_3/while/lstm_cell/split_1/split_dim:output:0=model_5/lstm_3/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_splitĆ
&model_5/lstm_3/while/lstm_cell/BiasAddBiasAdd/model_5/lstm_3/while/lstm_cell/MatMul:product:0/model_5/lstm_3/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ę
(model_5/lstm_3/while/lstm_cell/BiasAdd_1BiasAdd1model_5/lstm_3/while/lstm_cell/MatMul_1:product:0/model_5/lstm_3/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ę
(model_5/lstm_3/while/lstm_cell/BiasAdd_2BiasAdd1model_5/lstm_3/while/lstm_cell/MatMul_2:product:0/model_5/lstm_3/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ę
(model_5/lstm_3/while/lstm_cell/BiasAdd_3BiasAdd1model_5/lstm_3/while/lstm_cell/MatMul_3:product:0/model_5/lstm_3/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛
$model_5/lstm_3/while/lstm_cell/mul_4Mul"model_5_lstm_3_while_placeholder_2.model_5/lstm_3/while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛
$model_5/lstm_3/while/lstm_cell/mul_5Mul"model_5_lstm_3_while_placeholder_2.model_5/lstm_3/while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛
$model_5/lstm_3/while/lstm_cell/mul_6Mul"model_5_lstm_3_while_placeholder_2.model_5/lstm_3/while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˛
$model_5/lstm_3/while/lstm_cell/mul_7Mul"model_5_lstm_3_while_placeholder_2.model_5/lstm_3/while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
-model_5/lstm_3/while/lstm_cell/ReadVariableOpReadVariableOp8model_5_lstm_3_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
2model_5/lstm_3/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
4model_5/lstm_3/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
4model_5/lstm_3/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
,model_5/lstm_3/while/lstm_cell/strided_sliceStridedSlice5model_5/lstm_3/while/lstm_cell/ReadVariableOp:value:0;model_5/lstm_3/while/lstm_cell/strided_slice/stack:output:0=model_5/lstm_3/while/lstm_cell/strided_slice/stack_1:output:0=model_5/lstm_3/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskĹ
'model_5/lstm_3/while/lstm_cell/MatMul_4MatMul(model_5/lstm_3/while/lstm_cell/mul_4:z:05model_5/lstm_3/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Â
"model_5/lstm_3/while/lstm_cell/addAddV2/model_5/lstm_3/while/lstm_cell/BiasAdd:output:01model_5/lstm_3/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
&model_5/lstm_3/while/lstm_cell/SigmoidSigmoid&model_5/lstm_3/while/lstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
/model_5/lstm_3/while/lstm_cell/ReadVariableOp_1ReadVariableOp8model_5_lstm_3_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
4model_5/lstm_3/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
6model_5/lstm_3/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
6model_5/lstm_3/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
.model_5/lstm_3/while/lstm_cell/strided_slice_1StridedSlice7model_5/lstm_3/while/lstm_cell/ReadVariableOp_1:value:0=model_5/lstm_3/while/lstm_cell/strided_slice_1/stack:output:0?model_5/lstm_3/while/lstm_cell/strided_slice_1/stack_1:output:0?model_5/lstm_3/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÇ
'model_5/lstm_3/while/lstm_cell/MatMul_5MatMul(model_5/lstm_3/while/lstm_cell/mul_5:z:07model_5/lstm_3/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ć
$model_5/lstm_3/while/lstm_cell/add_1AddV21model_5/lstm_3/while/lstm_cell/BiasAdd_1:output:01model_5/lstm_3/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(model_5/lstm_3/while/lstm_cell/Sigmoid_1Sigmoid(model_5/lstm_3/while/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙°
$model_5/lstm_3/while/lstm_cell/mul_8Mul,model_5/lstm_3/while/lstm_cell/Sigmoid_1:y:0"model_5_lstm_3_while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
/model_5/lstm_3/while/lstm_cell/ReadVariableOp_2ReadVariableOp8model_5_lstm_3_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
4model_5/lstm_3/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
6model_5/lstm_3/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
6model_5/lstm_3/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
.model_5/lstm_3/while/lstm_cell/strided_slice_2StridedSlice7model_5/lstm_3/while/lstm_cell/ReadVariableOp_2:value:0=model_5/lstm_3/while/lstm_cell/strided_slice_2/stack:output:0?model_5/lstm_3/while/lstm_cell/strided_slice_2/stack_1:output:0?model_5/lstm_3/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÇ
'model_5/lstm_3/while/lstm_cell/MatMul_6MatMul(model_5/lstm_3/while/lstm_cell/mul_6:z:07model_5/lstm_3/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ć
$model_5/lstm_3/while/lstm_cell/add_2AddV21model_5/lstm_3/while/lstm_cell/BiasAdd_2:output:01model_5/lstm_3/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
#model_5/lstm_3/while/lstm_cell/TanhTanh(model_5/lstm_3/while/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ł
$model_5/lstm_3/while/lstm_cell/mul_9Mul*model_5/lstm_3/while/lstm_cell/Sigmoid:y:0'model_5/lstm_3/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
$model_5/lstm_3/while/lstm_cell/add_3AddV2(model_5/lstm_3/while/lstm_cell/mul_8:z:0(model_5/lstm_3/while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
/model_5/lstm_3/while/lstm_cell/ReadVariableOp_3ReadVariableOp8model_5_lstm_3_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0
4model_5/lstm_3/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       
6model_5/lstm_3/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
6model_5/lstm_3/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
.model_5/lstm_3/while/lstm_cell/strided_slice_3StridedSlice7model_5/lstm_3/while/lstm_cell/ReadVariableOp_3:value:0=model_5/lstm_3/while/lstm_cell/strided_slice_3/stack:output:0?model_5/lstm_3/while/lstm_cell/strided_slice_3/stack_1:output:0?model_5/lstm_3/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskÇ
'model_5/lstm_3/while/lstm_cell/MatMul_7MatMul(model_5/lstm_3/while/lstm_cell/mul_7:z:07model_5/lstm_3/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ć
$model_5/lstm_3/while/lstm_cell/add_4AddV21model_5/lstm_3/while/lstm_cell/BiasAdd_3:output:01model_5/lstm_3/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
(model_5/lstm_3/while/lstm_cell/Sigmoid_2Sigmoid(model_5/lstm_3/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
%model_5/lstm_3/while/lstm_cell/Tanh_1Tanh(model_5/lstm_3/while/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
%model_5/lstm_3/while/lstm_cell/mul_10Mul,model_5/lstm_3/while/lstm_cell/Sigmoid_2:y:0)model_5/lstm_3/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˙
9model_5/lstm_3/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"model_5_lstm_3_while_placeholder_1 model_5_lstm_3_while_placeholder)model_5/lstm_3/while/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŇ\
model_5/lstm_3/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
model_5/lstm_3/while/addAddV2 model_5_lstm_3_while_placeholder#model_5/lstm_3/while/add/y:output:0*
T0*
_output_shapes
: ^
model_5/lstm_3/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ł
model_5/lstm_3/while/add_1AddV26model_5_lstm_3_while_model_5_lstm_3_while_loop_counter%model_5/lstm_3/while/add_1/y:output:0*
T0*
_output_shapes
: 
model_5/lstm_3/while/IdentityIdentitymodel_5/lstm_3/while/add_1:z:0^model_5/lstm_3/while/NoOp*
T0*
_output_shapes
: Ś
model_5/lstm_3/while/Identity_1Identity<model_5_lstm_3_while_model_5_lstm_3_while_maximum_iterations^model_5/lstm_3/while/NoOp*
T0*
_output_shapes
: 
model_5/lstm_3/while/Identity_2Identitymodel_5/lstm_3/while/add:z:0^model_5/lstm_3/while/NoOp*
T0*
_output_shapes
: ł
model_5/lstm_3/while/Identity_3IdentityImodel_5/lstm_3/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^model_5/lstm_3/while/NoOp*
T0*
_output_shapes
: Ľ
model_5/lstm_3/while/Identity_4Identity)model_5/lstm_3/while/lstm_cell/mul_10:z:0^model_5/lstm_3/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¤
model_5/lstm_3/while/Identity_5Identity(model_5/lstm_3/while/lstm_cell/add_3:z:0^model_5/lstm_3/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ë
model_5/lstm_3/while/NoOpNoOp.^model_5/lstm_3/while/lstm_cell/ReadVariableOp0^model_5/lstm_3/while/lstm_cell/ReadVariableOp_10^model_5/lstm_3/while/lstm_cell/ReadVariableOp_20^model_5/lstm_3/while/lstm_cell/ReadVariableOp_34^model_5/lstm_3/while/lstm_cell/split/ReadVariableOp6^model_5/lstm_3/while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 "G
model_5_lstm_3_while_identity&model_5/lstm_3/while/Identity:output:0"K
model_5_lstm_3_while_identity_1(model_5/lstm_3/while/Identity_1:output:0"K
model_5_lstm_3_while_identity_2(model_5/lstm_3/while/Identity_2:output:0"K
model_5_lstm_3_while_identity_3(model_5/lstm_3/while/Identity_3:output:0"K
model_5_lstm_3_while_identity_4(model_5/lstm_3/while/Identity_4:output:0"K
model_5_lstm_3_while_identity_5(model_5/lstm_3/while/Identity_5:output:0"r
6model_5_lstm_3_while_lstm_cell_readvariableop_resource8model_5_lstm_3_while_lstm_cell_readvariableop_resource_0"
>model_5_lstm_3_while_lstm_cell_split_1_readvariableop_resource@model_5_lstm_3_while_lstm_cell_split_1_readvariableop_resource_0"~
<model_5_lstm_3_while_lstm_cell_split_readvariableop_resource>model_5_lstm_3_while_lstm_cell_split_readvariableop_resource_0"h
1model_5_lstm_3_while_model_5_lstm_3_strided_slice3model_5_lstm_3_while_model_5_lstm_3_strided_slice_0"ä
omodel_5_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_5_lstm_3_tensorarrayunstack_tensorlistfromtensorqmodel_5_lstm_3_while_tensorarrayv2read_tensorlistgetitem_model_5_lstm_3_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2^
-model_5/lstm_3/while/lstm_cell/ReadVariableOp-model_5/lstm_3/while/lstm_cell/ReadVariableOp2b
/model_5/lstm_3/while/lstm_cell/ReadVariableOp_1/model_5/lstm_3/while/lstm_cell/ReadVariableOp_12b
/model_5/lstm_3/while/lstm_cell/ReadVariableOp_2/model_5/lstm_3/while/lstm_cell/ReadVariableOp_22b
/model_5/lstm_3/while/lstm_cell/ReadVariableOp_3/model_5/lstm_3/while/lstm_cell/ReadVariableOp_32j
3model_5/lstm_3/while/lstm_cell/split/ReadVariableOp3model_5/lstm_3/while/lstm_cell/split/ReadVariableOp2n
5model_5/lstm_3/while/lstm_cell/split_1/ReadVariableOp5model_5/lstm_3/while/lstm_cell/split_1/ReadVariableOp:Y U

_output_shapes
: 
;
_user_specified_name#!model_5/lstm_3/while/loop_counter:_[

_output_shapes
: 
A
_user_specified_name)'model_5/lstm_3/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:TP

_output_shapes
: 
6
_user_specified_namemodel_5/lstm_3/strided_slice:nj

_output_shapes
: 
P
_user_specified_name86model_5/lstm_3/TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
Á	
ž
while_cond_52532
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_52532___redundant_placeholder03
/while_while_cond_52532___redundant_placeholder13
/while_while_cond_52532___redundant_placeholder23
/while_while_cond_52532___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
ą
ż
*model_5_attention_layer_while_1_cond_52065P
Lmodel_5_attention_layer_while_1_model_5_attention_layer_while_1_loop_counterV
Rmodel_5_attention_layer_while_1_model_5_attention_layer_while_1_maximum_iterations/
+model_5_attention_layer_while_1_placeholder1
-model_5_attention_layer_while_1_placeholder_11
-model_5_attention_layer_while_1_placeholder_2P
Lmodel_5_attention_layer_while_1_less_model_5_attention_layer_strided_slice_3g
cmodel_5_attention_layer_while_1_model_5_attention_layer_while_1_cond_52065___redundant_placeholder0g
cmodel_5_attention_layer_while_1_model_5_attention_layer_while_1_cond_52065___redundant_placeholder1,
(model_5_attention_layer_while_1_identity
Č
$model_5/attention_layer/while_1/LessLess+model_5_attention_layer_while_1_placeholderLmodel_5_attention_layer_while_1_less_model_5_attention_layer_strided_slice_3*
T0*
_output_shapes
: 
(model_5/attention_layer/while_1/IdentityIdentity(model_5/attention_layer/while_1/Less:z:0*
T0
*
_output_shapes
: "]
(model_5_attention_layer_while_1_identity1model_5/attention_layer/while_1/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :˙˙˙˙˙˙˙˙˙: :::d `

_output_shapes
: 
F
_user_specified_name.,model_5/attention_layer/while_1/loop_counter:jf

_output_shapes
: 
L
_user_specified_name42model_5/attention_layer/while_1/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:_[

_output_shapes
: 
A
_user_specified_name)'model_5/attention_layer/strided_slice_3:

_output_shapes
::

_output_shapes
:
ł˝
˙
while_body_54634
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
˘while/lstm_cell/ReadVariableOp˘ while/lstm_cell/ReadVariableOp_1˘ while/lstm_cell/ReadVariableOp_2˘ while/lstm_cell/ReadVariableOp_3˘$while/lstm_cell/split/ReadVariableOp˘&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0
while/lstm_cell/ones_likeOnesLike0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
while/lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
while/lstm_cell/dropout/MulMulwhile/lstm_cell/ones_like:y:0&while/lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙x
while/lstm_cell/dropout/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎ­
4while/lstm_cell/dropout/random_uniform/RandomUniformRandomUniform&while/lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0k
&while/lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>×
$while/lstm_cell/dropout/GreaterEqualGreaterEqual=while/lstm_cell/dropout/random_uniform/RandomUniform:output:0/while/lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ô
 while/lstm_cell/dropout/SelectV2SelectV2(while/lstm_cell/dropout/GreaterEqual:z:0while/lstm_cell/dropout/Mul:z:0(while/lstm_cell/dropout/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ? 
while/lstm_cell/dropout_1/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/lstm_cell/dropout_1/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ý
&while/lstm_cell/dropout_1/GreaterEqualGreaterEqual?while/lstm_cell/dropout_1/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_1/SelectV2SelectV2*while/lstm_cell/dropout_1/GreaterEqual:z:0!while/lstm_cell/dropout_1/Mul:z:0*while/lstm_cell/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ? 
while/lstm_cell/dropout_2/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/lstm_cell/dropout_2/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ý
&while/lstm_cell/dropout_2/GreaterEqualGreaterEqual?while/lstm_cell/dropout_2/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_2/SelectV2SelectV2*while/lstm_cell/dropout_2/GreaterEqual:z:0!while/lstm_cell/dropout_2/Mul:z:0*while/lstm_cell/dropout_2/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ? 
while/lstm_cell/dropout_3/MulMulwhile/lstm_cell/ones_like:y:0(while/lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
while/lstm_cell/dropout_3/ShapeShapewhile/lstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ý
&while/lstm_cell/dropout_3/GreaterEqualGreaterEqual?while/lstm_cell/dropout_3/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_3/SelectV2SelectV2*while/lstm_cell/dropout_3/GreaterEqual:z:0!while/lstm_cell/dropout_3/Mul:z:0*while/lstm_cell/dropout_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
while/lstm_cell/ones_like_1OnesLikewhile_placeholder_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
while/lstm_cell/dropout_4/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
while/lstm_cell/dropout_4/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell/dropout_4/GreaterEqualGreaterEqual?while/lstm_cell/dropout_4/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_4/SelectV2SelectV2*while/lstm_cell/dropout_4/GreaterEqual:z:0!while/lstm_cell/dropout_4/Mul:z:0*while/lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
while/lstm_cell/dropout_5/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
while/lstm_cell/dropout_5/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell/dropout_5/GreaterEqualGreaterEqual?while/lstm_cell/dropout_5/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_5/SelectV2SelectV2*while/lstm_cell/dropout_5/GreaterEqual:z:0!while/lstm_cell/dropout_5/Mul:z:0*while/lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
while/lstm_cell/dropout_6/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
while/lstm_cell/dropout_6/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell/dropout_6/GreaterEqualGreaterEqual?while/lstm_cell/dropout_6/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_6/SelectV2SelectV2*while/lstm_cell/dropout_6/GreaterEqual:z:0!while/lstm_cell/dropout_6/Mul:z:0*while/lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
while/lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?˘
while/lstm_cell/dropout_7/MulMulwhile/lstm_cell/ones_like_1:y:0(while/lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
while/lstm_cell/dropout_7/ShapeShapewhile/lstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎą
6while/lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform(while/lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0m
(while/lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ý
&while/lstm_cell/dropout_7/GreaterEqualGreaterEqual?while/lstm_cell/dropout_7/random_uniform/RandomUniform:output:01while/lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
!while/lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ü
"while/lstm_cell/dropout_7/SelectV2SelectV2*while/lstm_cell/dropout_7/GreaterEqual:z:0!while/lstm_cell/dropout_7/Mul:z:0*while/lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0)while/lstm_cell/dropout/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ž
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_1/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ž
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_2/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ž
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0+while/lstm_cell/dropout_3/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ö
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Č
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_4Mulwhile_placeholder_2+while/lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_5Mulwhile_placeholder_2+while/lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_6Mulwhile_placeholder_2+while/lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_7Mulwhile_placeholder_2+while/lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ˝
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
žI
ň
while_body_55148
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_shape_inputs_0_0;
'while_shape_1_readvariableop_resource_0:
<
(while_matmul_1_readvariableop_resource_0:
:
'while_shape_3_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_shape_inputs_09
%while_shape_1_readvariableop_resource:
:
&while_matmul_1_readvariableop_resource:
8
%while_shape_3_readvariableop_resource:	˘while/MatMul_1/ReadVariableOp˘while/transpose/ReadVariableOp˘ while/transpose_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0_
while/ShapeShapewhile_shape_inputs_0_0*
T0*
_output_shapes
::íĎ]
while/unstackUnpackwhile/Shape:output:0*
T0*
_output_shapes
: : : *	
num
while/Shape_1/ReadVariableOpReadVariableOp'while_shape_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0^
while/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      _
while/unstack_1Unpackwhile/Shape_1:output:0*
T0*
_output_shapes
: : *	
numd
while/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
while/ReshapeReshapewhile_shape_inputs_0_0while/Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/transpose/ReadVariableOpReadVariableOp'while_shape_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0e
while/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       
while/transpose	Transpose&while/transpose/ReadVariableOp:value:0while/transpose/perm:output:0*
T0* 
_output_shapes
:
f
while/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙z
while/Reshape_1Reshapewhile/transpose:y:0while/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
{
while/MatMulMatMulwhile/Reshape:output:0while/Reshape_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
while/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :Z
while/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :§
while/Reshape_2/shapePackwhile/unstack:output:0 while/Reshape_2/shape/1:output:0 while/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:
while/Reshape_2Reshapewhile/MatMul:product:0while/Reshape_2/shape:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1/ReadVariableOpReadVariableOp(while_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¤
while/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
while/ExpandDims
ExpandDimswhile/MatMul_1:product:0while/ExpandDims/dim:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
	while/addAddV2while/Reshape_2:output:0while/ExpandDims:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Y

while/TanhTanhwhile/add:z:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
while/Shape_2Shapewhile/Tanh:y:0*
T0*
_output_shapes
::íĎa
while/unstack_2Unpackwhile/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num
while/Shape_3/ReadVariableOpReadVariableOp'while_shape_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0^
while/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      _
while/unstack_3Unpackwhile/Shape_3:output:0*
T0*
_output_shapes
: : *	
numf
while/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   }
while/Reshape_3Reshapewhile/Tanh:y:0while/Reshape_3/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/transpose_1/ReadVariableOpReadVariableOp'while_shape_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0g
while/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
while/transpose_1	Transpose(while/transpose_1/ReadVariableOp:value:0while/transpose_1/perm:output:0*
T0*
_output_shapes
:	f
while/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙{
while/Reshape_4Reshapewhile/transpose_1:y:0while/Reshape_4/shape:output:0*
T0*
_output_shapes
:	~
while/MatMul_2MatMulwhile/Reshape_3:output:0while/Reshape_4:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
while/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value
B :Y
while/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Š
while/Reshape_5/shapePackwhile/unstack_2:output:0 while/Reshape_5/shape/1:output:0 while/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:
while/Reshape_5Reshapewhile/MatMul_2:product:0while/Reshape_5/shape:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/SqueezeSqueezewhile/Reshape_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims

˙˙˙˙˙˙˙˙˙c
while/SoftmaxSoftmaxwhile/Squeeze:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/Softmax:softmax:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_1AddV2while_placeholderwhile/add_1/y:output:0*
T0*
_output_shapes
: O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_2AddV2while_while_loop_counterwhile/add_2/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_2:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/Softmax:softmax:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

while/NoOpNoOp^while/MatMul_1/ReadVariableOp^while/transpose/ReadVariableOp!^while/transpose_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"R
&while_matmul_1_readvariableop_resource(while_matmul_1_readvariableop_resource_0"P
%while_shape_1_readvariableop_resource'while_shape_1_readvariableop_resource_0"P
%while_shape_3_readvariableop_resource'while_shape_3_readvariableop_resource_0".
while_shape_inputs_0while_shape_inputs_0_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙: : : 2>
while/MatMul_1/ReadVariableOpwhile/MatMul_1/ReadVariableOp2@
while/transpose/ReadVariableOpwhile/transpose/ReadVariableOp2D
 while/transpose_1/ReadVariableOp while/transpose_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:WS
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
Ť
¤
F__inference_embedding_1_layer_call_and_return_conditional_losses_52682

inputs*
embedding_lookup_52677:
×8
identity˘embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ă
embedding_lookupResourceGatherembedding_lookup_52677Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/52677*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:%!

_user_specified_name52677
Ť}
î
A__inference_lstm_3_layer_call_and_return_conditional_losses_54476
inputs_0;
'lstm_cell_split_readvariableop_resource:
8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity

identity_1

identity_2˘lstm_cell/ReadVariableOp˘lstm_cell/ReadVariableOp_1˘lstm_cell/ReadVariableOp_2˘lstm_cell/ReadVariableOp_3˘lstm_cell/split/ReadVariableOp˘ lstm_cell/split_1/ReadVariableOp˘whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::íĎ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::íĎ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskl
lstm_cell/ones_likeOnesLikestrided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell/ones_like_1OnesLikezeros:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell/mulMulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/mul_1Mulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/mul_2Mulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/mul_3Mulstrided_slice_2:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype0Ä
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitz
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ś
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_cell/mul_4Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_cell/mul_5Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_cell/mul_6Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_cell/mul_7Mulzeros:output:0lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ô
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_54344*
condR
while_cond_54343*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕ
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
÷
×
&__inference_lstm_3_layer_call_fn_53825
inputs_0
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1

identity_2˘StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_52411}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0:%!

_user_specified_name53813:%!

_user_specified_name53815:%!

_user_specified_name53817

m
A__inference_concat_layer_call_and_return_conditional_losses_55351
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙e
IdentityIdentityconcat:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0:_[
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1
´
ü
B__inference_dense_1_layer_call_and_return_conditional_losses_55391

inputs5
!tensordot_readvariableop_resource:
×8.
biasadd_readvariableop_resource:	×8
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
×8*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::íĎY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙×8\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:×8Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:×8*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8d
SoftmaxSoftmaxBiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8n
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
/
÷
*model_5_attention_layer_while_1_body_52066P
Lmodel_5_attention_layer_while_1_model_5_attention_layer_while_1_loop_counterV
Rmodel_5_attention_layer_while_1_model_5_attention_layer_while_1_maximum_iterations/
+model_5_attention_layer_while_1_placeholder1
-model_5_attention_layer_while_1_placeholder_11
-model_5_attention_layer_while_1_placeholder_2M
Imodel_5_attention_layer_while_1_model_5_attention_layer_strided_slice_3_0
model_5_attention_layer_while_1_tensorarrayv2read_tensorlistgetitem_model_5_attention_layer_tensorarrayunstack_1_tensorlistfromtensor_01
-model_5_attention_layer_while_1_mul_input_8_0,
(model_5_attention_layer_while_1_identity.
*model_5_attention_layer_while_1_identity_1.
*model_5_attention_layer_while_1_identity_2.
*model_5_attention_layer_while_1_identity_3.
*model_5_attention_layer_while_1_identity_4K
Gmodel_5_attention_layer_while_1_model_5_attention_layer_strided_slice_3
model_5_attention_layer_while_1_tensorarrayv2read_tensorlistgetitem_model_5_attention_layer_tensorarrayunstack_1_tensorlistfromtensor/
+model_5_attention_layer_while_1_mul_input_8˘
Qmodel_5/attention_layer/while_1/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  Ş
Cmodel_5/attention_layer/while_1/TensorArrayV2Read/TensorListGetItemTensorListGetItemmodel_5_attention_layer_while_1_tensorarrayv2read_tensorlistgetitem_model_5_attention_layer_tensorarrayunstack_1_tensorlistfromtensor_0+model_5_attention_layer_while_1_placeholderZmodel_5/attention_layer/while_1/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0y
.model_5/attention_layer/while_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ô
*model_5/attention_layer/while_1/ExpandDims
ExpandDimsJmodel_5/attention_layer/while_1/TensorArrayV2Read/TensorListGetItem:item:07model_5/attention_layer/while_1/ExpandDims/dim:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙Ć
#model_5/attention_layer/while_1/mulMul-model_5_attention_layer_while_1_mul_input_8_03model_5/attention_layer/while_1/ExpandDims:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙w
5model_5/attention_layer/while_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ć
#model_5/attention_layer/while_1/SumSum'model_5/attention_layer/while_1/mul:z:0>model_5/attention_layer/while_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ł
Dmodel_5/attention_layer/while_1/TensorArrayV2Write/TensorListSetItemTensorListSetItem-model_5_attention_layer_while_1_placeholder_1+model_5_attention_layer_while_1_placeholder,model_5/attention_layer/while_1/Sum:output:0*
_output_shapes
: *
element_dtype0:éčŇg
%model_5/attention_layer/while_1/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ş
#model_5/attention_layer/while_1/addAddV2+model_5_attention_layer_while_1_placeholder.model_5/attention_layer/while_1/add/y:output:0*
T0*
_output_shapes
: i
'model_5/attention_layer/while_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ď
%model_5/attention_layer/while_1/add_1AddV2Lmodel_5_attention_layer_while_1_model_5_attention_layer_while_1_loop_counter0model_5/attention_layer/while_1/add_1/y:output:0*
T0*
_output_shapes
: 
(model_5/attention_layer/while_1/IdentityIdentity)model_5/attention_layer/while_1/add_1:z:0*
T0*
_output_shapes
: Ť
*model_5/attention_layer/while_1/Identity_1IdentityRmodel_5_attention_layer_while_1_model_5_attention_layer_while_1_maximum_iterations*
T0*
_output_shapes
: 
*model_5/attention_layer/while_1/Identity_2Identity'model_5/attention_layer/while_1/add:z:0*
T0*
_output_shapes
: ­
*model_5/attention_layer/while_1/Identity_3IdentityTmodel_5/attention_layer/while_1/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 
*model_5/attention_layer/while_1/Identity_4Identity,model_5/attention_layer/while_1/Sum:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"]
(model_5_attention_layer_while_1_identity1model_5/attention_layer/while_1/Identity:output:0"a
*model_5_attention_layer_while_1_identity_13model_5/attention_layer/while_1/Identity_1:output:0"a
*model_5_attention_layer_while_1_identity_23model_5/attention_layer/while_1/Identity_2:output:0"a
*model_5_attention_layer_while_1_identity_33model_5/attention_layer/while_1/Identity_3:output:0"a
*model_5_attention_layer_while_1_identity_43model_5/attention_layer/while_1/Identity_4:output:0"
Gmodel_5_attention_layer_while_1_model_5_attention_layer_strided_slice_3Imodel_5_attention_layer_while_1_model_5_attention_layer_strided_slice_3_0"\
+model_5_attention_layer_while_1_mul_input_8-model_5_attention_layer_while_1_mul_input_8_0"
model_5_attention_layer_while_1_tensorarrayv2read_tensorlistgetitem_model_5_attention_layer_tensorarrayunstack_1_tensorlistfromtensormodel_5_attention_layer_while_1_tensorarrayv2read_tensorlistgetitem_model_5_attention_layer_tensorarrayunstack_1_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙:d `

_output_shapes
: 
F
_user_specified_name.,model_5/attention_layer/while_1/loop_counter:jf

_output_shapes
: 
L
_user_specified_name42model_5/attention_layer/while_1/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:_[

_output_shapes
: 
A
_user_specified_name)'model_5/attention_layer/strided_slice_3:yu

_output_shapes
: 
[
_user_specified_nameCAmodel_5/attention_layer/TensorArrayUnstack_1/TensorListFromTensor:VR
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8
­/

!__inference__traced_restore_55762
file_prefix;
'assignvariableop_embedding_1_embeddings:
×8:
&assignvariableop_1_attention_layer_w_a:
:
&assignvariableop_2_attention_layer_u_a:
9
&assignvariableop_3_attention_layer_v_a:	5
!assignvariableop_4_dense_1_kernel:
×8.
assignvariableop_5_dense_1_bias:	×8>
*assignvariableop_6_lstm_3_lstm_cell_kernel:
H
4assignvariableop_7_lstm_3_lstm_cell_recurrent_kernel:
7
(assignvariableop_8_lstm_3_lstm_cell_bias:	
identity_10˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8Î
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*ô
valueęBç
B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/W_a/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/U_a/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/V_a/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B Đ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ş
AssignVariableOpAssignVariableOp'assignvariableop_embedding_1_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:˝
AssignVariableOp_1AssignVariableOp&assignvariableop_1_attention_layer_w_aIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:˝
AssignVariableOp_2AssignVariableOp&assignvariableop_2_attention_layer_u_aIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:˝
AssignVariableOp_3AssignVariableOp&assignvariableop_3_attention_layer_v_aIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:¸
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ś
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Á
AssignVariableOp_6AssignVariableOp*assignvariableop_6_lstm_3_lstm_cell_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ë
AssignVariableOp_7AssignVariableOp4assignvariableop_7_lstm_3_lstm_cell_recurrent_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:ż
AssignVariableOp_8AssignVariableOp(assignvariableop_8_lstm_3_lstm_cell_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 

Identity_9Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^NoOp"/device:CPU:0*
T0*
_output_shapes
: V
Identity_10IdentityIdentity_9:output:0^NoOp_1*
T0*
_output_shapes
: ß
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8*
_output_shapes
 "#
identity_10Identity_10:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
: : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_8:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:62
0
_user_specified_nameembedding_1/embeddings:3/
-
_user_specified_nameattention_layer/W_a:3/
-
_user_specified_nameattention_layer/U_a:3/
-
_user_specified_nameattention_layer/V_a:.*
(
_user_specified_namedense_1/kernel:,(
&
_user_specified_namedense_1/bias:73
1
_user_specified_namelstm_3/lstm_cell/kernel:A=
;
_user_specified_name#!lstm_3/lstm_cell/recurrent_kernel:5	1
/
_user_specified_namelstm_3/lstm_cell/bias
Ĺ
ô
)__inference_lstm_cell_layer_call_fn_55408

inputs
states_0
states_1
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1

identity_2˘StatefulPartitionedCallŞ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_52326p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_1:%!

_user_specified_name55396:%!

_user_specified_name55398:%!

_user_specified_name55400
óA
§
D__inference_lstm_cell_layer_call_and_return_conditional_losses_52519

inputs

states
states_11
split_readvariableop_resource:
.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpP
	ones_likeOnesLikeinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
ones_like_1OnesLikestates*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
mulMulinputsones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_1Mulinputsones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_2Mulinputsones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_3Mulinputsones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype0Ś
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_4Mulstatesones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_5Mulstatesones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_6Mulstatesones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_7Mulstatesones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
B
Š
D__inference_lstm_cell_layer_call_and_return_conditional_losses_55645

inputs
states_0
states_11
split_readvariableop_resource:
.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpP
	ones_likeOnesLikeinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
ones_like_1OnesLikestates_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
mulMulinputsones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_1Mulinputsones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_2Mulinputsones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_3Mulinputsones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype0Ś
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_4Mulstates_0ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_5Mulstates_0ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_6Mulstates_0ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
mul_7Mulstates_0ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ŕ
×
while_1_cond_53257 
while_1_while_1_loop_counter&
"while_1_while_1_maximum_iterations
while_1_placeholder
while_1_placeholder_1
while_1_placeholder_2 
while_1_less_strided_slice_37
3while_1_while_1_cond_53257___redundant_placeholder07
3while_1_while_1_cond_53257___redundant_placeholder1
while_1_identity
h
while_1/LessLesswhile_1_placeholderwhile_1_less_strided_slice_3*
T0*
_output_shapes
: O
while_1/IdentityIdentitywhile_1/Less:z:0*
T0
*
_output_shapes
: "-
while_1_identitywhile_1/Identity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: : : : :˙˙˙˙˙˙˙˙˙: :::L H

_output_shapes
: 
.
_user_specified_namewhile_1/loop_counter:RN

_output_shapes
: 
4
_user_specified_namewhile_1/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_3:

_output_shapes
::

_output_shapes
:
Ô

+__inference_embedding_1_layer_call_fn_53801

inputs
unknown:
×8
identity˘StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_52682}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:%!

_user_specified_name53797
ť	
ź
while_cond_54923
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_54923___redundant_placeholder03
/while_while_cond_54923___redundant_placeholder13
/while_while_cond_54923___redundant_placeholder23
/while_while_cond_54923___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
:
ť~
§
D__inference_lstm_cell_layer_call_and_return_conditional_losses_52326

inputs

states
states_11
split_readvariableop_resource:
.
split_1_readvariableop_resource:	+
readvariableop_resource:

identity

identity_1

identity_2˘ReadVariableOp˘ReadVariableOp_1˘ReadVariableOp_2˘ReadVariableOp_3˘split/ReadVariableOp˘split_1/ReadVariableOpP
	ones_likeOnesLikeinputs*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?l
dropout/MulMulones_like:y:0dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
dropout/ShapeShapeones_like:y:0*
T0*
_output_shapes
::íĎ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?p
dropout_1/MulMulones_like:y:0dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
dropout_1/ShapeShapeones_like:y:0*
T0*
_output_shapes
::íĎ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>­
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_1/SelectV2SelectV2dropout_1/GreaterEqual:z:0dropout_1/Mul:z:0dropout_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?p
dropout_2/MulMulones_like:y:0dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
dropout_2/ShapeShapeones_like:y:0*
T0*
_output_shapes
::íĎ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>­
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_2/SelectV2SelectV2dropout_2/GreaterEqual:z:0dropout_2/Mul:z:0dropout_2/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?p
dropout_3/MulMulones_like:y:0dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
dropout_3/ShapeShapeones_like:y:0*
T0*
_output_shapes
::íĎ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>­
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_3/SelectV2SelectV2dropout_3/GreaterEqual:z:0dropout_3/Mul:z:0dropout_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
ones_like_1OnesLikestates*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?r
dropout_4/MulMulones_like_1:y:0dropout_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
dropout_4/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::íĎ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_4/SelectV2SelectV2dropout_4/GreaterEqual:z:0dropout_4/Mul:z:0dropout_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?r
dropout_5/MulMulones_like_1:y:0dropout_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
dropout_5/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::íĎ
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_5/SelectV2SelectV2dropout_5/GreaterEqual:z:0dropout_5/Mul:z:0dropout_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?r
dropout_6/MulMulones_like_1:y:0dropout_6/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
dropout_6/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::íĎ
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_6/SelectV2SelectV2dropout_6/GreaterEqual:z:0dropout_6/Mul:z:0dropout_6/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?r
dropout_7/MulMulones_like_1:y:0dropout_7/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
dropout_7/ShapeShapeones_like_1:y:0*
T0*
_output_shapes
::íĎ
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>­
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout_7/SelectV2SelectV2dropout_7/GreaterEqual:z:0dropout_7/Mul:z:0dropout_7/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
mulMulinputsdropout/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
mul_1Mulinputsdropout_1/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
mul_2Mulinputsdropout_2/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
mul_3Mulinputsdropout_3/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :t
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
*
dtype0Ś
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split\
MatMulMatMulmul:z:0split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : s
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_spliti
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙m
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
mul_4Mulstatesdropout_4/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
mul_5Mulstatesdropout_5/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
mul_6Mulstatesdropout_6/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
mul_7Mulstatesdropout_7/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskh
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
SigmoidSigmoidadd:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_1Sigmoid	add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙X
mul_8MulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙J
TanhTanh	add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
mul_9MulSigmoid:y:0Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙W
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÷
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskj
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙i
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R
	Sigmoid_2Sigmoid	add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙L
Tanh_1Tanh	add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
mul_10MulSigmoid_2:y:0
Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
IdentityIdentity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\

Identity_1Identity
mul_10:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[

Identity_2Identity	add_3:z:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates:PL
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_namestates:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ż$
Ë
while_body_52533
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_52557_0:
&
while_lstm_cell_52559_0:	+
while_lstm_cell_52561_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_52557:
$
while_lstm_cell_52559:	)
while_lstm_cell_52561:
˘'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0Ś
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_52557_0while_lstm_cell_52559_0while_lstm_cell_52561_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_52519Ů
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_52557while_lstm_cell_52557_0"0
while_lstm_cell_52559while_lstm_cell_52559_0"0
while_lstm_cell_52561while_lstm_cell_52561_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:%!

_user_specified_name52557:%	!

_user_specified_name52559:%
!

_user_specified_name52561
Éť

A__inference_lstm_3_layer_call_and_return_conditional_losses_53039

inputs
initial_state
initial_state_1;
'lstm_cell_split_readvariableop_resource:
8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity

identity_1

identity_2˘lstm_cell/ReadVariableOp˘lstm_cell/ReadVariableOp_1˘lstm_cell/ReadVariableOp_2˘lstm_cell/ReadVariableOp_3˘lstm_cell/split/ReadVariableOp˘ lstm_cell/split_1/ReadVariableOp˘whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::íĎ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskl
lstm_cell/ones_likeOnesLikestrided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
lstm_cell/dropout/MulMullstm_cell/ones_like:y:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎĄ
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ĺ
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ź
lstm_cell/dropout/SelectV2SelectV2"lstm_cell/dropout/GreaterEqual:z:0lstm_cell/dropout/Mul:z:0"lstm_cell/dropout/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ë
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_1/SelectV2SelectV2$lstm_cell/dropout_1/GreaterEqual:z:0lstm_cell/dropout_1/Mul:z:0$lstm_cell/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ë
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_2/SelectV2SelectV2$lstm_cell/dropout_2/GreaterEqual:z:0lstm_cell/dropout_2/Mul:z:0$lstm_cell/dropout_2/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ë
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_3/SelectV2SelectV2$lstm_cell/dropout_3/GreaterEqual:z:0lstm_cell/dropout_3/Mul:z:0$lstm_cell/dropout_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell/ones_like_1OnesLikeinitial_state*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_4/SelectV2SelectV2$lstm_cell/dropout_4/GreaterEqual:z:0lstm_cell/dropout_4/Mul:z:0$lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_5/SelectV2SelectV2$lstm_cell/dropout_5/GreaterEqual:z:0lstm_cell/dropout_5/Mul:z:0$lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_6/SelectV2SelectV2$lstm_cell/dropout_6/GreaterEqual:z:0lstm_cell/dropout_6/Mul:z:0$lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_7/SelectV2SelectV2$lstm_cell/dropout_7/GreaterEqual:z:0lstm_cell/dropout_7/Mul:z:0$lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mulMulstrided_slice_1:output:0#lstm_cell/dropout/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_1Mulstrided_slice_1:output:0%lstm_cell/dropout_1/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_2Mulstrided_slice_1:output:0%lstm_cell/dropout_2/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_3Mulstrided_slice_1:output:0%lstm_cell/dropout_3/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype0Ä
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitz
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ś
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_4Mulinitial_state%lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_5Mulinitial_state%lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_6Mulinitial_state%lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_7Mulinitial_state%lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0initial_state_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Đ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_stateinitial_state_1strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_52843*
condR
while_cond_52842*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕ
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:WS
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameinitial_state:WS
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameinitial_state:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
÷
×
&__inference_lstm_3_layer_call_fn_53840
inputs_0
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1

identity_2˘StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_52604}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0:%!

_user_specified_name53828:%!

_user_specified_name53830:%!

_user_specified_name53832
ˇ	
Ř
while_cond_53125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice3
/while_while_cond_53125___redundant_placeholder03
/while_while_cond_53125___redundant_placeholder13
/while_while_cond_53125___redundant_placeholder23
/while_while_cond_53125___redundant_placeholder33
/while_while_cond_53125___redundant_placeholder4
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :˙˙˙˙˙˙˙˙˙: ::::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
::

_output_shapes
:
é
˛
'__inference_model_5_layer_call_fn_53693
input_2
input_8
input_6
input_7
unknown:
×8
	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:
×8
	unknown_7:	×8
identity

identity_1

identity_2˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2input_8input_6input_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_5_layer_call_and_return_conditional_losses_53633}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2:VR
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_6:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_7:%!

_user_specified_name53669:%!

_user_specified_name53671:%!

_user_specified_name53673:%!

_user_specified_name53675:%!

_user_specified_name53677:%	!

_user_specified_name53679:%
!

_user_specified_name53681:%!

_user_specified_name53683:%!

_user_specified_name53685
Č
É
while_1_body_53258 
while_1_while_1_loop_counter&
"while_1_while_1_maximum_iterations
while_1_placeholder
while_1_placeholder_1
while_1_placeholder_2
while_1_strided_slice_3_0[
Wwhile_1_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0
while_1_mul_inputs_0
while_1_identity
while_1_identity_1
while_1_identity_2
while_1_identity_3
while_1_identity_4
while_1_strided_slice_3Y
Uwhile_1_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor
while_1_mul_inputs
9while_1/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  ą
+while_1/TensorArrayV2Read/TensorListGetItemTensorListGetItemWwhile_1_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_1_placeholderBwhile_1/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0a
while_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
while_1/ExpandDims
ExpandDims2while_1/TensorArrayV2Read/TensorListGetItem:item:0while_1/ExpandDims/dim:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙}
while_1/mulMulwhile_1_mul_inputs_0while_1/ExpandDims:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :~
while_1/SumSumwhile_1/mul:z:0&while_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
,while_1/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_1_placeholder_1while_1_placeholderwhile_1/Sum:output:0*
_output_shapes
: *
element_dtype0:éčŇO
while_1/add/yConst*
_output_shapes
: *
dtype0*
value	B :b
while_1/addAddV2while_1_placeholderwhile_1/add/y:output:0*
T0*
_output_shapes
: Q
while_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :o
while_1/add_1AddV2while_1_while_1_loop_counterwhile_1/add_1/y:output:0*
T0*
_output_shapes
: P
while_1/IdentityIdentitywhile_1/add_1:z:0*
T0*
_output_shapes
: c
while_1/Identity_1Identity"while_1_while_1_maximum_iterations*
T0*
_output_shapes
: P
while_1/Identity_2Identitywhile_1/add:z:0*
T0*
_output_shapes
: }
while_1/Identity_3Identity<while_1/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: g
while_1/Identity_4Identitywhile_1/Sum:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"-
while_1_identitywhile_1/Identity:output:0"1
while_1_identity_1while_1/Identity_1:output:0"1
while_1_identity_2while_1/Identity_2:output:0"1
while_1_identity_3while_1/Identity_3:output:0"1
while_1_identity_4while_1/Identity_4:output:0"*
while_1_mul_inputswhile_1_mul_inputs_0"4
while_1_strided_slice_3while_1_strided_slice_3_0"°
Uwhile_1_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_1_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙:L H

_output_shapes
: 
.
_user_specified_namewhile_1/loop_counter:RN

_output_shapes
: 
4
_user_specified_namewhile_1/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_3:a]

_output_shapes
: 
C
_user_specified_name+)TensorArrayUnstack_1/TensorListFromTensor:UQ
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs

č
model_5_lstm_3_while_cond_51730:
6model_5_lstm_3_while_model_5_lstm_3_while_loop_counter@
<model_5_lstm_3_while_model_5_lstm_3_while_maximum_iterations$
 model_5_lstm_3_while_placeholder&
"model_5_lstm_3_while_placeholder_1&
"model_5_lstm_3_while_placeholder_2&
"model_5_lstm_3_while_placeholder_3:
6model_5_lstm_3_while_less_model_5_lstm_3_strided_sliceQ
Mmodel_5_lstm_3_while_model_5_lstm_3_while_cond_51730___redundant_placeholder0Q
Mmodel_5_lstm_3_while_model_5_lstm_3_while_cond_51730___redundant_placeholder1Q
Mmodel_5_lstm_3_while_model_5_lstm_3_while_cond_51730___redundant_placeholder2Q
Mmodel_5_lstm_3_while_model_5_lstm_3_while_cond_51730___redundant_placeholder3!
model_5_lstm_3_while_identity

model_5/lstm_3/while/LessLess model_5_lstm_3_while_placeholder6model_5_lstm_3_while_less_model_5_lstm_3_strided_slice*
T0*
_output_shapes
: i
model_5/lstm_3/while/IdentityIdentitymodel_5/lstm_3/while/Less:z:0*
T0
*
_output_shapes
: "G
model_5_lstm_3_while_identity&model_5/lstm_3/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::Y U

_output_shapes
: 
;
_user_specified_name#!model_5/lstm_3/while/loop_counter:_[

_output_shapes
: 
A
_user_specified_name)'model_5/lstm_3/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:TP

_output_shapes
: 
6
_user_specified_namemodel_5/lstm_3/strided_slice:

_output_shapes
:
¸
Đ
(model_5_attention_layer_while_cond_51933L
Hmodel_5_attention_layer_while_model_5_attention_layer_while_loop_counterR
Nmodel_5_attention_layer_while_model_5_attention_layer_while_maximum_iterations-
)model_5_attention_layer_while_placeholder/
+model_5_attention_layer_while_placeholder_1/
+model_5_attention_layer_while_placeholder_2L
Hmodel_5_attention_layer_while_less_model_5_attention_layer_strided_slicec
_model_5_attention_layer_while_model_5_attention_layer_while_cond_51933___redundant_placeholder0c
_model_5_attention_layer_while_model_5_attention_layer_while_cond_51933___redundant_placeholder1c
_model_5_attention_layer_while_model_5_attention_layer_while_cond_51933___redundant_placeholder2c
_model_5_attention_layer_while_model_5_attention_layer_while_cond_51933___redundant_placeholder3c
_model_5_attention_layer_while_model_5_attention_layer_while_cond_51933___redundant_placeholder4*
&model_5_attention_layer_while_identity
Ŕ
"model_5/attention_layer/while/LessLess)model_5_attention_layer_while_placeholderHmodel_5_attention_layer_while_less_model_5_attention_layer_strided_slice*
T0*
_output_shapes
: {
&model_5/attention_layer/while/IdentityIdentity&model_5/attention_layer/while/Less:z:0*
T0
*
_output_shapes
: "Y
&model_5_attention_layer_while_identity/model_5/attention_layer/while/Identity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2: : : : :˙˙˙˙˙˙˙˙˙: ::::::b ^

_output_shapes
: 
D
_user_specified_name,*model_5/attention_layer/while/loop_counter:hd

_output_shapes
: 
J
_user_specified_name20model_5/attention_layer/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:]Y

_output_shapes
: 
?
_user_specified_name'%model_5/attention_layer/strided_slice:

_output_shapes
::

_output_shapes
:
´
ü
B__inference_dense_1_layer_call_and_return_conditional_losses_53363

inputs5
!tensordot_readvariableop_resource:
×8.
biasadd_readvariableop_resource:	×8
identity˘BiasAdd/ReadVariableOp˘Tensordot/ReadVariableOp|
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
×8*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::íĎY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ť
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ż
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙×8\
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:×8Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:×8*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8d
SoftmaxSoftmaxBiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8n
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
é:

A__inference_lstm_3_layer_call_and_return_conditional_losses_52411

inputs#
lstm_cell_52327:

lstm_cell_52329:	#
lstm_cell_52331:

identity

identity_1

identity_2˘!lstm_cell/StatefulPartitionedCall˘whileI
ShapeShapeinputs*
T0*
_output_shapes
::íĎ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::íĎ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskč
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_52327lstm_cell_52329lstm_cell_52331*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_52326n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_52327lstm_cell_52329lstm_cell_52331*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_52340*
condR
while_cond_52339*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙N
NoOpNoOp"^lstm_cell/StatefulPartitionedCall^while*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:%!

_user_specified_name52327:%!

_user_specified_name52329:%!

_user_specified_name52331
Ô
Í
while_1_body_55280 
while_1_while_1_loop_counter&
"while_1_while_1_maximum_iterations
while_1_placeholder
while_1_placeholder_1
while_1_placeholder_2
while_1_strided_slice_3_0[
Wwhile_1_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0
while_1_mul_inputs_0_0
while_1_identity
while_1_identity_1
while_1_identity_2
while_1_identity_3
while_1_identity_4
while_1_strided_slice_3Y
Uwhile_1_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor
while_1_mul_inputs_0
9while_1/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  ą
+while_1/TensorArrayV2Read/TensorListGetItemTensorListGetItemWwhile_1_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0while_1_placeholderBwhile_1/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0a
while_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
while_1/ExpandDims
ExpandDims2while_1/TensorArrayV2Read/TensorListGetItem:item:0while_1/ExpandDims/dim:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
while_1/mulMulwhile_1_mul_inputs_0_0while_1/ExpandDims:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙_
while_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :~
while_1/SumSumwhile_1/mul:z:0&while_1/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
,while_1/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_1_placeholder_1while_1_placeholderwhile_1/Sum:output:0*
_output_shapes
: *
element_dtype0:éčŇO
while_1/add/yConst*
_output_shapes
: *
dtype0*
value	B :b
while_1/addAddV2while_1_placeholderwhile_1/add/y:output:0*
T0*
_output_shapes
: Q
while_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :o
while_1/add_1AddV2while_1_while_1_loop_counterwhile_1/add_1/y:output:0*
T0*
_output_shapes
: P
while_1/IdentityIdentitywhile_1/add_1:z:0*
T0*
_output_shapes
: c
while_1/Identity_1Identity"while_1_while_1_maximum_iterations*
T0*
_output_shapes
: P
while_1/Identity_2Identitywhile_1/add:z:0*
T0*
_output_shapes
: }
while_1/Identity_3Identity<while_1/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: g
while_1/Identity_4Identitywhile_1/Sum:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙"-
while_1_identitywhile_1/Identity:output:0"1
while_1_identity_1while_1/Identity_1:output:0"1
while_1_identity_2while_1/Identity_2:output:0"1
while_1_identity_3while_1/Identity_3:output:0"1
while_1_identity_4while_1/Identity_4:output:0".
while_1_mul_inputs_0while_1_mul_inputs_0_0"4
while_1_strided_slice_3while_1_strided_slice_3_0"°
Uwhile_1_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensorWwhile_1_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_1_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙:L H

_output_shapes
: 
.
_user_specified_namewhile_1/loop_counter:RN

_output_shapes
: 
4
_user_specified_namewhile_1/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_3:a]

_output_shapes
: 
C
_user_specified_name+)TensorArrayUnstack_1/TensorListFromTensor:WS
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0
ßť

A__inference_lstm_3_layer_call_and_return_conditional_losses_54830

inputs
initial_state_0
initial_state_1;
'lstm_cell_split_readvariableop_resource:
8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity

identity_1

identity_2˘lstm_cell/ReadVariableOp˘lstm_cell/ReadVariableOp_1˘lstm_cell/ReadVariableOp_2˘lstm_cell/ReadVariableOp_3˘lstm_cell/split/ReadVariableOp˘ lstm_cell/split_1/ReadVariableOp˘whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::íĎ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskl
lstm_cell/ones_likeOnesLikestrided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
lstm_cell/dropout/MulMullstm_cell/ones_like:y:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎĄ
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ĺ
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ź
lstm_cell/dropout/SelectV2SelectV2"lstm_cell/dropout/GreaterEqual:z:0lstm_cell/dropout/Mul:z:0"lstm_cell/dropout/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ë
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_1/SelectV2SelectV2$lstm_cell/dropout_1/GreaterEqual:z:0lstm_cell/dropout_1/Mul:z:0$lstm_cell/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ë
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_2/SelectV2SelectV2$lstm_cell/dropout_2/GreaterEqual:z:0lstm_cell/dropout_2/Mul:z:0$lstm_cell/dropout_2/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ë
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_3/SelectV2SelectV2$lstm_cell/dropout_3/GreaterEqual:z:0lstm_cell/dropout_3/Mul:z:0$lstm_cell/dropout_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
lstm_cell/ones_like_1OnesLikeinitial_state_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_4/SelectV2SelectV2$lstm_cell/dropout_4/GreaterEqual:z:0lstm_cell/dropout_4/Mul:z:0$lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_5/SelectV2SelectV2$lstm_cell/dropout_5/GreaterEqual:z:0lstm_cell/dropout_5/Mul:z:0$lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_6/SelectV2SelectV2$lstm_cell/dropout_6/GreaterEqual:z:0lstm_cell/dropout_6/Mul:z:0$lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_7/SelectV2SelectV2$lstm_cell/dropout_7/GreaterEqual:z:0lstm_cell/dropout_7/Mul:z:0$lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mulMulstrided_slice_1:output:0#lstm_cell/dropout/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_1Mulstrided_slice_1:output:0%lstm_cell/dropout_1/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_2Mulstrided_slice_1:output:0%lstm_cell/dropout_2/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_3Mulstrided_slice_1:output:0%lstm_cell/dropout_3/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype0Ä
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitz
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ś
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_4Mulinitial_state_0%lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_5Mulinitial_state_0%lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_6Mulinitial_state_0%lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_7Mulinitial_state_0%lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0initial_state_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ň
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_state_0initial_state_1strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_54634*
condR
while_cond_54633*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕ
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:YU
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinitial_state_0:YU
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinitial_state_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Á	
ž
while_cond_52339
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_52339___redundant_placeholder03
/while_while_cond_52339___redundant_placeholder13
/while_while_cond_52339___redundant_placeholder23
/while_while_cond_52339___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
ť	
ź
while_cond_52842
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_52842___redundant_placeholder03
/while_while_cond_52842___redundant_placeholder13
/while_while_cond_52842___redundant_placeholder23
/while_while_cond_52842___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
:
Ę$

B__inference_model_5_layer_call_and_return_conditional_losses_53372
input_2
input_8
input_6
input_7%
embedding_1_52683:
×8 
lstm_3_53040:

lstm_3_53042:	 
lstm_3_53044:
)
attention_layer_53317:
)
attention_layer_53319:
(
attention_layer_53321:	!
dense_1_53364:
×8
dense_1_53366:	×8
identity

identity_1

identity_2˘'attention_layer/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘#embedding_1/StatefulPartitionedCall˘lstm_3/StatefulPartitionedCallö
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_52683*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_52682ę
lstm_3/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0input_6input_7lstm_3_53040lstm_3_53042lstm_3_53044*
Tin

2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_53039
'attention_layer/StatefulPartitionedCallStatefulPartitionedCallinput_8'lstm_3/StatefulPartitionedCall:output:0attention_layer_53317attention_layer_53319attention_layer_53321*
Tin	
2*
Tout
2*
_collective_manager_ids
 *V
_output_shapesD
B:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_attention_layer_layer_call_and_return_conditional_losses_53316
concat/PartitionedCallPartitionedCall'lstm_3/StatefulPartitionedCall:output:00attention_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_53331
dense_1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0dense_1_53364dense_1_53366*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_53363
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8y

Identity_1Identity'lstm_3/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y

Identity_2Identity'lstm_3/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
NoOpNoOp(^attention_layer/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : 2R
'attention_layer/StatefulPartitionedCall'attention_layer/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2:VR
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_6:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_7:%!

_user_specified_name52683:%!

_user_specified_name53040:%!

_user_specified_name53042:%!

_user_specified_name53044:%!

_user_specified_name53317:%	!

_user_specified_name53319:%
!

_user_specified_name53321:%!

_user_specified_name53364:%!

_user_specified_name53366
Á	
ž
while_cond_54343
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_54343___redundant_placeholder03
/while_while_cond_54343___redundant_placeholder13
/while_while_cond_54343___redundant_placeholder23
/while_while_cond_54343___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
ÓÄ
î
A__inference_lstm_3_layer_call_and_return_conditional_losses_54239
inputs_0;
'lstm_cell_split_readvariableop_resource:
8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity

identity_1

identity_2˘lstm_cell/ReadVariableOp˘lstm_cell/ReadVariableOp_1˘lstm_cell/ReadVariableOp_2˘lstm_cell/ReadVariableOp_3˘lstm_cell/split/ReadVariableOp˘ lstm_cell/split_1/ReadVariableOp˘whileK
ShapeShapeinputs_0*
T0*
_output_shapes
::íĎ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    m
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙S
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    s
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙R
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
::íĎ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskl
lstm_cell/ones_likeOnesLikestrided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙\
lstm_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
lstm_cell/dropout/MulMullstm_cell/ones_like:y:0 lstm_cell/dropout/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
lstm_cell/dropout/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎĄ
.lstm_cell/dropout/random_uniform/RandomUniformRandomUniform lstm_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0e
 lstm_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ĺ
lstm_cell/dropout/GreaterEqualGreaterEqual7lstm_cell/dropout/random_uniform/RandomUniform:output:0)lstm_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    ź
lstm_cell/dropout/SelectV2SelectV2"lstm_cell/dropout/GreaterEqual:z:0lstm_cell/dropout/Mul:z:0"lstm_cell/dropout/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
lstm_cell/dropout_1/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
lstm_cell/dropout_1/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_1/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ë
 lstm_cell/dropout_1/GreaterEqualGreaterEqual9lstm_cell/dropout_1/random_uniform/RandomUniform:output:0+lstm_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_1/SelectV2SelectV2$lstm_cell/dropout_1/GreaterEqual:z:0lstm_cell/dropout_1/Mul:z:0$lstm_cell/dropout_1/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
lstm_cell/dropout_2/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
lstm_cell/dropout_2/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_2/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ë
 lstm_cell/dropout_2/GreaterEqualGreaterEqual9lstm_cell/dropout_2/random_uniform/RandomUniform:output:0+lstm_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_2/SelectV2SelectV2$lstm_cell/dropout_2/GreaterEqual:z:0lstm_cell/dropout_2/Mul:z:0$lstm_cell/dropout_2/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUŐ?
lstm_cell/dropout_3/MulMullstm_cell/ones_like:y:0"lstm_cell/dropout_3/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
lstm_cell/dropout_3/ShapeShapelstm_cell/ones_like:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_3/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_3/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚĚ>Ë
 lstm_cell/dropout_3/GreaterEqualGreaterEqual9lstm_cell/dropout_3/random_uniform/RandomUniform:output:0+lstm_cell/dropout_3/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_3/SelectV2SelectV2$lstm_cell/dropout_3/GreaterEqual:z:0lstm_cell/dropout_3/Mul:z:0$lstm_cell/dropout_3/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙d
lstm_cell/ones_like_1OnesLikezeros:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_4/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_4/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_cell/dropout_4/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_4/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_4/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell/dropout_4/GreaterEqualGreaterEqual9lstm_cell/dropout_4/random_uniform/RandomUniform:output:0+lstm_cell/dropout_4/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_4/SelectV2SelectV2$lstm_cell/dropout_4/GreaterEqual:z:0lstm_cell/dropout_4/Mul:z:0$lstm_cell/dropout_4/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_5/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_5/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_cell/dropout_5/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_5/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_5/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell/dropout_5/GreaterEqualGreaterEqual9lstm_cell/dropout_5/random_uniform/RandomUniform:output:0+lstm_cell/dropout_5/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_5/SelectV2SelectV2$lstm_cell/dropout_5/GreaterEqual:z:0lstm_cell/dropout_5/Mul:z:0$lstm_cell/dropout_5/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_6/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_6/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_cell/dropout_6/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_6/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_6/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell/dropout_6/GreaterEqualGreaterEqual9lstm_cell/dropout_6/random_uniform/RandomUniform:output:0+lstm_cell/dropout_6/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_6/SelectV2SelectV2$lstm_cell/dropout_6/GreaterEqual:z:0lstm_cell/dropout_6/Mul:z:0$lstm_cell/dropout_6/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
lstm_cell/dropout_7/MulMullstm_cell/ones_like_1:y:0"lstm_cell/dropout_7/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙p
lstm_cell/dropout_7/ShapeShapelstm_cell/ones_like_1:y:0*
T0*
_output_shapes
::íĎĽ
0lstm_cell/dropout_7/random_uniform/RandomUniformRandomUniform"lstm_cell/dropout_7/Shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0g
"lstm_cell/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍĚL>Ë
 lstm_cell/dropout_7/GreaterEqualGreaterEqual9lstm_cell/dropout_7/random_uniform/RandomUniform:output:0+lstm_cell/dropout_7/GreaterEqual/y:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/dropout_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ä
lstm_cell/dropout_7/SelectV2SelectV2$lstm_cell/dropout_7/GreaterEqual:z:0lstm_cell/dropout_7/Mul:z:0$lstm_cell/dropout_7/Const_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mulMulstrided_slice_2:output:0#lstm_cell/dropout/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_1Mulstrided_slice_2:output:0%lstm_cell/dropout_1/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_2Mulstrided_slice_2:output:0%lstm_cell/dropout_2/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_3Mulstrided_slice_2:output:0%lstm_cell/dropout_3/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype0Ä
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitz
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ś
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_4Mulzeros:output:0%lstm_cell/dropout_4/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_5Mulzeros:output:0%lstm_cell/dropout_5/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_6Mulzeros:output:0%lstm_cell/dropout_6/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/mul_7Mulzeros:output:0%lstm_cell/dropout_7/SelectV2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ô
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_54043*
condR
while_cond_54042*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕ
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ż$
Ë
while_body_52340
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
while_lstm_cell_52364_0:
&
while_lstm_cell_52366_0:	+
while_lstm_cell_52368_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
while_lstm_cell_52364:
$
while_lstm_cell_52366:	)
while_lstm_cell_52368:
˘'while/lstm_cell/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0Ś
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_52364_0while_lstm_cell_52366_0while_lstm_cell_52368_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_52326Ů
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙R

while/NoOpNoOp(^while/lstm_cell/StatefulPartitionedCall*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_52364while_lstm_cell_52364_0"0
while_lstm_cell_52366while_lstm_cell_52366_0"0
while_lstm_cell_52368while_lstm_cell_52368_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:%!

_user_specified_name52364:%	!

_user_specified_name52366:%
!

_user_specified_name52368
Îŕ
î

 __inference__wrapped_model_52154
input_2
input_8
input_6
input_7>
*model_5_embedding_1_embedding_lookup_51638:
×8J
6model_5_lstm_3_lstm_cell_split_readvariableop_resource:
G
8model_5_lstm_3_lstm_cell_split_1_readvariableop_resource:	D
0model_5_lstm_3_lstm_cell_readvariableop_resource:
K
7model_5_attention_layer_shape_2_readvariableop_resource:
L
8model_5_attention_layer_matmul_1_readvariableop_resource:
J
7model_5_attention_layer_shape_4_readvariableop_resource:	E
1model_5_dense_1_tensordot_readvariableop_resource:
×8>
/model_5_dense_1_biasadd_readvariableop_resource:	×8
identity

identity_1

identity_2˘/model_5/attention_layer/MatMul_1/ReadVariableOp˘2model_5/attention_layer/transpose_1/ReadVariableOp˘2model_5/attention_layer/transpose_2/ReadVariableOp˘model_5/attention_layer/while˘&model_5/dense_1/BiasAdd/ReadVariableOp˘(model_5/dense_1/Tensordot/ReadVariableOp˘$model_5/embedding_1/embedding_lookup˘'model_5/lstm_3/lstm_cell/ReadVariableOp˘)model_5/lstm_3/lstm_cell/ReadVariableOp_1˘)model_5/lstm_3/lstm_cell/ReadVariableOp_2˘)model_5/lstm_3/lstm_cell/ReadVariableOp_3˘-model_5/lstm_3/lstm_cell/split/ReadVariableOp˘/model_5/lstm_3/lstm_cell/split_1/ReadVariableOp˘model_5/lstm_3/whiles
model_5/embedding_1/CastCastinput_2*

DstT0*

SrcT0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
$model_5/embedding_1/embedding_lookupResourceGather*model_5_embedding_1_embedding_lookup_51638model_5/embedding_1/Cast:y:0*
Tindices0*=
_class3
1/loc:@model_5/embedding_1/embedding_lookup/51638*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0¨
-model_5/embedding_1/embedding_lookup/IdentityIdentity-model_5/embedding_1/embedding_lookup:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙r
model_5/lstm_3/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ĺ
model_5/lstm_3/transpose	Transpose6model_5/embedding_1/embedding_lookup/Identity:output:0&model_5/lstm_3/transpose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙n
model_5/lstm_3/ShapeShapemodel_5/lstm_3/transpose:y:0*
T0*
_output_shapes
::íĎl
"model_5/lstm_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$model_5/lstm_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$model_5/lstm_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
model_5/lstm_3/strided_sliceStridedSlicemodel_5/lstm_3/Shape:output:0+model_5/lstm_3/strided_slice/stack:output:0-model_5/lstm_3/strided_slice/stack_1:output:0-model_5/lstm_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
*model_5/lstm_3/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ß
model_5/lstm_3/TensorArrayV2TensorListReserve3model_5/lstm_3/TensorArrayV2/element_shape:output:0%model_5/lstm_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
Dmodel_5/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
6model_5/lstm_3/TensorArrayUnstack/TensorListFromTensorTensorListFromTensormodel_5/lstm_3/transpose:y:0Mmodel_5/lstm_3/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇn
$model_5/lstm_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&model_5/lstm_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&model_5/lstm_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
model_5/lstm_3/strided_slice_1StridedSlicemodel_5/lstm_3/transpose:y:0-model_5/lstm_3/strided_slice_1/stack:output:0/model_5/lstm_3/strided_slice_1/stack_1:output:0/model_5/lstm_3/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask
"model_5/lstm_3/lstm_cell/ones_likeOnesLike'model_5/lstm_3/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
$model_5/lstm_3/lstm_cell/ones_like_1OnesLikeinput_6*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙§
model_5/lstm_3/lstm_cell/mulMul'model_5/lstm_3/strided_slice_1:output:0&model_5/lstm_3/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
model_5/lstm_3/lstm_cell/mul_1Mul'model_5/lstm_3/strided_slice_1:output:0&model_5/lstm_3/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
model_5/lstm_3/lstm_cell/mul_2Mul'model_5/lstm_3/strided_slice_1:output:0&model_5/lstm_3/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
model_5/lstm_3/lstm_cell/mul_3Mul'model_5/lstm_3/strided_slice_1:output:0&model_5/lstm_3/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
(model_5/lstm_3/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ś
-model_5/lstm_3/lstm_cell/split/ReadVariableOpReadVariableOp6model_5_lstm_3_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype0ń
model_5/lstm_3/lstm_cell/splitSplit1model_5/lstm_3/lstm_cell/split/split_dim:output:05model_5/lstm_3/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split§
model_5/lstm_3/lstm_cell/MatMulMatMul model_5/lstm_3/lstm_cell/mul:z:0'model_5/lstm_3/lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
!model_5/lstm_3/lstm_cell/MatMul_1MatMul"model_5/lstm_3/lstm_cell/mul_1:z:0'model_5/lstm_3/lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
!model_5/lstm_3/lstm_cell/MatMul_2MatMul"model_5/lstm_3/lstm_cell/mul_2:z:0'model_5/lstm_3/lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
!model_5/lstm_3/lstm_cell/MatMul_3MatMul"model_5/lstm_3/lstm_cell/mul_3:z:0'model_5/lstm_3/lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
*model_5/lstm_3/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Ľ
/model_5/lstm_3/lstm_cell/split_1/ReadVariableOpReadVariableOp8model_5_lstm_3_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ă
 model_5/lstm_3/lstm_cell/split_1Split3model_5/lstm_3/lstm_cell/split_1/split_dim:output:07model_5/lstm_3/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split´
 model_5/lstm_3/lstm_cell/BiasAddBiasAdd)model_5/lstm_3/lstm_cell/MatMul:product:0)model_5/lstm_3/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"model_5/lstm_3/lstm_cell/BiasAdd_1BiasAdd+model_5/lstm_3/lstm_cell/MatMul_1:product:0)model_5/lstm_3/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"model_5/lstm_3/lstm_cell/BiasAdd_2BiasAdd+model_5/lstm_3/lstm_cell/MatMul_2:product:0)model_5/lstm_3/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
"model_5/lstm_3/lstm_cell/BiasAdd_3BiasAdd+model_5/lstm_3/lstm_cell/MatMul_3:product:0)model_5/lstm_3/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_5/lstm_3/lstm_cell/mul_4Mulinput_6(model_5/lstm_3/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_5/lstm_3/lstm_cell/mul_5Mulinput_6(model_5/lstm_3/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_5/lstm_3/lstm_cell/mul_6Mulinput_6(model_5/lstm_3/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_5/lstm_3/lstm_cell/mul_7Mulinput_6(model_5/lstm_3/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'model_5/lstm_3/lstm_cell/ReadVariableOpReadVariableOp0model_5_lstm_3_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0}
,model_5/lstm_3/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
.model_5/lstm_3/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
.model_5/lstm_3/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ę
&model_5/lstm_3/lstm_cell/strided_sliceStridedSlice/model_5/lstm_3/lstm_cell/ReadVariableOp:value:05model_5/lstm_3/lstm_cell/strided_slice/stack:output:07model_5/lstm_3/lstm_cell/strided_slice/stack_1:output:07model_5/lstm_3/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskł
!model_5/lstm_3/lstm_cell/MatMul_4MatMul"model_5/lstm_3/lstm_cell/mul_4:z:0/model_5/lstm_3/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙°
model_5/lstm_3/lstm_cell/addAddV2)model_5/lstm_3/lstm_cell/BiasAdd:output:0+model_5/lstm_3/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 model_5/lstm_3/lstm_cell/SigmoidSigmoid model_5/lstm_3/lstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)model_5/lstm_3/lstm_cell/ReadVariableOp_1ReadVariableOp0model_5_lstm_3_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
.model_5/lstm_3/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0model_5/lstm_3/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0model_5/lstm_3/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(model_5/lstm_3/lstm_cell/strided_slice_1StridedSlice1model_5/lstm_3/lstm_cell/ReadVariableOp_1:value:07model_5/lstm_3/lstm_cell/strided_slice_1/stack:output:09model_5/lstm_3/lstm_cell/strided_slice_1/stack_1:output:09model_5/lstm_3/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!model_5/lstm_3/lstm_cell/MatMul_5MatMul"model_5/lstm_3/lstm_cell/mul_5:z:01model_5/lstm_3/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
model_5/lstm_3/lstm_cell/add_1AddV2+model_5/lstm_3/lstm_cell/BiasAdd_1:output:0+model_5/lstm_3/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"model_5/lstm_3/lstm_cell/Sigmoid_1Sigmoid"model_5/lstm_3/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_5/lstm_3/lstm_cell/mul_8Mul&model_5/lstm_3/lstm_cell/Sigmoid_1:y:0input_7*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)model_5/lstm_3/lstm_cell/ReadVariableOp_2ReadVariableOp0model_5_lstm_3_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
.model_5/lstm_3/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0model_5/lstm_3/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0model_5/lstm_3/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(model_5/lstm_3/lstm_cell/strided_slice_2StridedSlice1model_5/lstm_3/lstm_cell/ReadVariableOp_2:value:07model_5/lstm_3/lstm_cell/strided_slice_2/stack:output:09model_5/lstm_3/lstm_cell/strided_slice_2/stack_1:output:09model_5/lstm_3/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!model_5/lstm_3/lstm_cell/MatMul_6MatMul"model_5/lstm_3/lstm_cell/mul_6:z:01model_5/lstm_3/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
model_5/lstm_3/lstm_cell/add_2AddV2+model_5/lstm_3/lstm_cell/BiasAdd_2:output:0+model_5/lstm_3/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
model_5/lstm_3/lstm_cell/TanhTanh"model_5/lstm_3/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
model_5/lstm_3/lstm_cell/mul_9Mul$model_5/lstm_3/lstm_cell/Sigmoid:y:0!model_5/lstm_3/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
model_5/lstm_3/lstm_cell/add_3AddV2"model_5/lstm_3/lstm_cell/mul_8:z:0"model_5/lstm_3/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)model_5/lstm_3/lstm_cell/ReadVariableOp_3ReadVariableOp0model_5_lstm_3_lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0
.model_5/lstm_3/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       
0model_5/lstm_3/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
0model_5/lstm_3/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ô
(model_5/lstm_3/lstm_cell/strided_slice_3StridedSlice1model_5/lstm_3/lstm_cell/ReadVariableOp_3:value:07model_5/lstm_3/lstm_cell/strided_slice_3/stack:output:09model_5/lstm_3/lstm_cell/strided_slice_3/stack_1:output:09model_5/lstm_3/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_maskľ
!model_5/lstm_3/lstm_cell/MatMul_7MatMul"model_5/lstm_3/lstm_cell/mul_7:z:01model_5/lstm_3/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙´
model_5/lstm_3/lstm_cell/add_4AddV2+model_5/lstm_3/lstm_cell/BiasAdd_3:output:0+model_5/lstm_3/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"model_5/lstm_3/lstm_cell/Sigmoid_2Sigmoid"model_5/lstm_3/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
model_5/lstm_3/lstm_cell/Tanh_1Tanh"model_5/lstm_3/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
model_5/lstm_3/lstm_cell/mul_10Mul&model_5/lstm_3/lstm_cell/Sigmoid_2:y:0#model_5/lstm_3/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙}
,model_5/lstm_3/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ă
model_5/lstm_3/TensorArrayV2_1TensorListReserve5model_5/lstm_3/TensorArrayV2_1/element_shape:output:0%model_5/lstm_3/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇU
model_5/lstm_3/timeConst*
_output_shapes
: *
dtype0*
value	B : r
'model_5/lstm_3/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙c
!model_5/lstm_3/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ö
model_5/lstm_3/whileWhile*model_5/lstm_3/while/loop_counter:output:00model_5/lstm_3/while/maximum_iterations:output:0model_5/lstm_3/time:output:0'model_5/lstm_3/TensorArrayV2_1:handle:0input_6input_7%model_5/lstm_3/strided_slice:output:0Fmodel_5/lstm_3/TensorArrayUnstack/TensorListFromTensor:output_handle:06model_5_lstm_3_lstm_cell_split_readvariableop_resource8model_5_lstm_3_lstm_cell_split_1_readvariableop_resource0model_5_lstm_3_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
model_5_lstm_3_while_body_51731*+
cond#R!
model_5_lstm_3_while_cond_51730*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
?model_5/lstm_3/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ů
1model_5/lstm_3/TensorArrayV2Stack/TensorListStackTensorListStackmodel_5/lstm_3/while:output:3Hmodel_5/lstm_3/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0w
$model_5/lstm_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙p
&model_5/lstm_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&model_5/lstm_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
model_5/lstm_3/strided_slice_2StridedSlice:model_5/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0-model_5/lstm_3/strided_slice_2/stack:output:0/model_5/lstm_3/strided_slice_2/stack_1:output:0/model_5/lstm_3/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskt
model_5/lstm_3/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Í
model_5/lstm_3/transpose_1	Transpose:model_5/lstm_3/TensorArrayV2Stack/TensorListStack:tensor:0(model_5/lstm_3/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙j
model_5/lstm_3/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    o
-model_5/attention_layer/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
model_5/attention_layer/SumSuminput_86model_5/attention_layer/Sum/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙q
/model_5/attention_layer/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
model_5/attention_layer/Sum_1Suminput_88model_5/attention_layer/Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
&model_5/attention_layer/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ż
!model_5/attention_layer/transpose	Transposemodel_5/lstm_3/transpose_1:y:0/model_5/attention_layer/transpose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
model_5/attention_layer/ShapeShape%model_5/attention_layer/transpose:y:0*
T0*
_output_shapes
::íĎu
+model_5/attention_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-model_5/attention_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-model_5/attention_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%model_5/attention_layer/strided_sliceStridedSlice&model_5/attention_layer/Shape:output:04model_5/attention_layer/strided_slice/stack:output:06model_5/attention_layer/strided_slice/stack_1:output:06model_5/attention_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
3model_5/attention_layer/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ú
%model_5/attention_layer/TensorArrayV2TensorListReserve<model_5/attention_layer/TensorArrayV2/element_shape:output:0.model_5/attention_layer/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
Mmodel_5/attention_layer/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¨
?model_5/attention_layer/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor%model_5/attention_layer/transpose:y:0Vmodel_5/attention_layer/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇw
-model_5/attention_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model_5/attention_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model_5/attention_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
'model_5/attention_layer/strided_slice_1StridedSlice%model_5/attention_layer/transpose:y:06model_5/attention_layer/strided_slice_1/stack:output:08model_5/attention_layer/strided_slice_1/stack_1:output:08model_5/attention_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskd
model_5/attention_layer/Shape_1Shapeinput_8*
T0*
_output_shapes
::íĎ
model_5/attention_layer/unstackUnpack(model_5/attention_layer/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num¨
.model_5/attention_layer/Shape_2/ReadVariableOpReadVariableOp7model_5_attention_layer_shape_2_readvariableop_resource* 
_output_shapes
:
*
dtype0p
model_5/attention_layer/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"      
!model_5/attention_layer/unstack_1Unpack(model_5/attention_layer/Shape_2:output:0*
T0*
_output_shapes
: : *	
numv
%model_5/attention_layer/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
model_5/attention_layer/ReshapeReshapeinput_8.model_5/attention_layer/Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
2model_5/attention_layer/transpose_1/ReadVariableOpReadVariableOp7model_5_attention_layer_shape_2_readvariableop_resource* 
_output_shapes
:
*
dtype0y
(model_5/attention_layer/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Ę
#model_5/attention_layer/transpose_1	Transpose:model_5/attention_layer/transpose_1/ReadVariableOp:value:01model_5/attention_layer/transpose_1/perm:output:0*
T0* 
_output_shapes
:
x
'model_5/attention_layer/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙˛
!model_5/attention_layer/Reshape_1Reshape'model_5/attention_layer/transpose_1:y:00model_5/attention_layer/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
ą
model_5/attention_layer/MatMulMatMul(model_5/attention_layer/Reshape:output:0*model_5/attention_layer/Reshape_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
)model_5/attention_layer/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :l
)model_5/attention_layer/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ď
'model_5/attention_layer/Reshape_2/shapePack(model_5/attention_layer/unstack:output:02model_5/attention_layer/Reshape_2/shape/1:output:02model_5/attention_layer/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:Ŕ
!model_5/attention_layer/Reshape_2Reshape(model_5/attention_layer/MatMul:product:00model_5/attention_layer/Reshape_2/shape:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
/model_5/attention_layer/MatMul_1/ReadVariableOpReadVariableOp8model_5_attention_layer_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0Č
 model_5/attention_layer/MatMul_1MatMul0model_5/attention_layer/strided_slice_1:output:07model_5/attention_layer/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙h
&model_5/attention_layer/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ä
"model_5/attention_layer/ExpandDims
ExpandDims*model_5/attention_layer/MatMul_1:product:0/model_5/attention_layer/ExpandDims/dim:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
model_5/attention_layer/addAddV2*model_5/attention_layer/Reshape_2:output:0+model_5/attention_layer/ExpandDims:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙}
model_5/attention_layer/TanhTanhmodel_5/attention_layer/add:z:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙}
model_5/attention_layer/Shape_3Shape model_5/attention_layer/Tanh:y:0*
T0*
_output_shapes
::íĎ
!model_5/attention_layer/unstack_2Unpack(model_5/attention_layer/Shape_3:output:0*
T0*
_output_shapes
: : : *	
num§
.model_5/attention_layer/Shape_4/ReadVariableOpReadVariableOp7model_5_attention_layer_shape_4_readvariableop_resource*
_output_shapes
:	*
dtype0p
model_5/attention_layer/Shape_4Const*
_output_shapes
:*
dtype0*
valueB"      
!model_5/attention_layer/unstack_3Unpack(model_5/attention_layer/Shape_4:output:0*
T0*
_output_shapes
: : *	
numx
'model_5/attention_layer/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ł
!model_5/attention_layer/Reshape_3Reshape model_5/attention_layer/Tanh:y:00model_5/attention_layer/Reshape_3/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
2model_5/attention_layer/transpose_2/ReadVariableOpReadVariableOp7model_5_attention_layer_shape_4_readvariableop_resource*
_output_shapes
:	*
dtype0y
(model_5/attention_layer/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       É
#model_5/attention_layer/transpose_2	Transpose:model_5/attention_layer/transpose_2/ReadVariableOp:value:01model_5/attention_layer/transpose_2/perm:output:0*
T0*
_output_shapes
:	x
'model_5/attention_layer/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙ą
!model_5/attention_layer/Reshape_4Reshape'model_5/attention_layer/transpose_2:y:00model_5/attention_layer/Reshape_4/shape:output:0*
T0*
_output_shapes
:	´
 model_5/attention_layer/MatMul_2MatMul*model_5/attention_layer/Reshape_3:output:0*model_5/attention_layer/Reshape_4:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
)model_5/attention_layer/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value
B :k
)model_5/attention_layer/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :ń
'model_5/attention_layer/Reshape_5/shapePack*model_5/attention_layer/unstack_2:output:02model_5/attention_layer/Reshape_5/shape/1:output:02model_5/attention_layer/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:Á
!model_5/attention_layer/Reshape_5Reshape*model_5/attention_layer/MatMul_2:product:00model_5/attention_layer/Reshape_5/shape:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙Š
model_5/attention_layer/SqueezeSqueeze*model_5/attention_layer/Reshape_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims

˙˙˙˙˙˙˙˙˙
model_5/attention_layer/SoftmaxSoftmax(model_5/attention_layer/Squeeze:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
5model_5/attention_layer/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  ţ
'model_5/attention_layer/TensorArrayV2_1TensorListReserve>model_5/attention_layer/TensorArrayV2_1/element_shape:output:0.model_5/attention_layer/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ^
model_5/attention_layer/timeConst*
_output_shapes
: *
dtype0*
value	B : {
0model_5/attention_layer/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙l
*model_5/attention_layer/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ř
model_5/attention_layer/whileWhile3model_5/attention_layer/while/loop_counter:output:09model_5/attention_layer/while/maximum_iterations:output:0%model_5/attention_layer/time:output:00model_5/attention_layer/TensorArrayV2_1:handle:0&model_5/attention_layer/Sum_1:output:0.model_5/attention_layer/strided_slice:output:0Omodel_5/attention_layer/TensorArrayUnstack/TensorListFromTensor:output_handle:0input_87model_5_attention_layer_shape_2_readvariableop_resource8model_5_attention_layer_matmul_1_readvariableop_resource7model_5_attention_layer_shape_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙: : : *%
_read_only_resource_inputs
	
*4
body,R*
(model_5_attention_layer_while_body_51934*4
cond,R*
(model_5_attention_layer_while_cond_51933*R
output_shapesA
?: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙: : : *
parallel_iterations 
Hmodel_5/attention_layer/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  
:model_5/attention_layer/TensorArrayV2Stack/TensorListStackTensorListStack&model_5/attention_layer/while:output:3Qmodel_5/attention_layer/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0
-model_5/attention_layer/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙y
/model_5/attention_layer/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/model_5/attention_layer/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
'model_5/attention_layer/strided_slice_2StridedSliceCmodel_5/attention_layer/TensorArrayV2Stack/TensorListStack:tensor:06model_5/attention_layer/strided_slice_2/stack:output:08model_5/attention_layer/strided_slice_2/stack_1:output:08model_5/attention_layer/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask}
(model_5/attention_layer/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          č
#model_5/attention_layer/transpose_3	TransposeCmodel_5/attention_layer/TensorArrayV2Stack/TensorListStack:tensor:01model_5/attention_layer/transpose_3/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙}
(model_5/attention_layer/transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ě
#model_5/attention_layer/transpose_4	Transpose'model_5/attention_layer/transpose_3:y:01model_5/attention_layer/transpose_4/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
model_5/attention_layer/Shape_5Shape'model_5/attention_layer/transpose_4:y:0*
T0*
_output_shapes
::íĎw
-model_5/attention_layer/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model_5/attention_layer/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model_5/attention_layer/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'model_5/attention_layer/strided_slice_3StridedSlice(model_5/attention_layer/Shape_5:output:06model_5/attention_layer/strided_slice_3/stack:output:08model_5/attention_layer/strided_slice_3/stack_1:output:08model_5/attention_layer/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
5model_5/attention_layer/TensorArrayV2_3/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
'model_5/attention_layer/TensorArrayV2_3TensorListReserve>model_5/attention_layer/TensorArrayV2_3/element_shape:output:00model_5/attention_layer/strided_slice_3:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ 
Omodel_5/attention_layer/TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  Ž
Amodel_5/attention_layer/TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensor'model_5/attention_layer/transpose_4:y:0Xmodel_5/attention_layer/TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇw
-model_5/attention_layer/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/model_5/attention_layer/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/model_5/attention_layer/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ä
'model_5/attention_layer/strided_slice_4StridedSlice'model_5/attention_layer/transpose_4:y:06model_5/attention_layer/strided_slice_4/stack:output:08model_5/attention_layer/strided_slice_4/stack_1:output:08model_5/attention_layer/strided_slice_4/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_masks
(model_5/attention_layer/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Î
$model_5/attention_layer/ExpandDims_1
ExpandDims0model_5/attention_layer/strided_slice_4:output:01model_5/attention_layer/ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
model_5/attention_layer/mulMulinput_8-model_5/attention_layer/ExpandDims_1:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙q
/model_5/attention_layer/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :˛
model_5/attention_layer/Sum_2Summodel_5/attention_layer/mul:z:08model_5/attention_layer/Sum_2/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
5model_5/attention_layer/TensorArrayV2_4/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
'model_5/attention_layer/TensorArrayV2_4TensorListReserve>model_5/attention_layer/TensorArrayV2_4/element_shape:output:00model_5/attention_layer/strided_slice_3:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ`
model_5/attention_layer/time_1Const*
_output_shapes
: *
dtype0*
value	B : }
2model_5/attention_layer/while_1/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙n
,model_5/attention_layer/while_1/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ď
model_5/attention_layer/while_1StatelessWhile5model_5/attention_layer/while_1/loop_counter:output:0;model_5/attention_layer/while_1/maximum_iterations:output:0'model_5/attention_layer/time_1:output:00model_5/attention_layer/TensorArrayV2_4:handle:0$model_5/attention_layer/Sum:output:00model_5/attention_layer/strided_slice_3:output:0Qmodel_5/attention_layer/TensorArrayUnstack_1/TensorListFromTensor:output_handle:0input_8*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*M
_output_shapes;
9: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *6
body.R,
*model_5_attention_layer_while_1_body_52066*6
cond.R,
*model_5_attention_layer_while_1_cond_52065*L
output_shapes;
9: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙*
parallel_iterations 
Jmodel_5/attention_layer/TensorArrayV2Stack_1/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
<model_5/attention_layer/TensorArrayV2Stack_1/TensorListStackTensorListStack(model_5/attention_layer/while_1:output:3Smodel_5/attention_layer/TensorArrayV2Stack_1/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0
-model_5/attention_layer/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙y
/model_5/attention_layer/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/model_5/attention_layer/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
'model_5/attention_layer/strided_slice_5StridedSliceEmodel_5/attention_layer/TensorArrayV2Stack_1/TensorListStack:tensor:06model_5/attention_layer/strided_slice_5/stack:output:08model_5/attention_layer/strided_slice_5/stack_1:output:08model_5/attention_layer/strided_slice_5/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask}
(model_5/attention_layer/transpose_5/permConst*
_output_shapes
:*
dtype0*!
valueB"          ę
#model_5/attention_layer/transpose_5	TransposeEmodel_5/attention_layer/TensorArrayV2Stack_1/TensorListStack:tensor:01model_5/attention_layer/transpose_5/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙\
model_5/concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ř
model_5/concat/concatConcatV2model_5/lstm_3/transpose_1:y:0'model_5/attention_layer/transpose_5:y:0#model_5/concat/concat/axis:output:0*
N*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
(model_5/dense_1/Tensordot/ReadVariableOpReadVariableOp1model_5_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
×8*
dtype0h
model_5/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:o
model_5/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
model_5/dense_1/Tensordot/ShapeShapemodel_5/concat/concat:output:0*
T0*
_output_shapes
::íĎi
'model_5/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ű
"model_5/dense_1/Tensordot/GatherV2GatherV2(model_5/dense_1/Tensordot/Shape:output:0'model_5/dense_1/Tensordot/free:output:00model_5/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
)model_5/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ˙
$model_5/dense_1/Tensordot/GatherV2_1GatherV2(model_5/dense_1/Tensordot/Shape:output:0'model_5/dense_1/Tensordot/axes:output:02model_5/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
model_5/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model_5/dense_1/Tensordot/ProdProd+model_5/dense_1/Tensordot/GatherV2:output:0(model_5/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: k
!model_5/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¤
 model_5/dense_1/Tensordot/Prod_1Prod-model_5/dense_1/Tensordot/GatherV2_1:output:0*model_5/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: g
%model_5/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ü
 model_5/dense_1/Tensordot/concatConcatV2'model_5/dense_1/Tensordot/free:output:0'model_5/dense_1/Tensordot/axes:output:0.model_5/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Š
model_5/dense_1/Tensordot/stackPack'model_5/dense_1/Tensordot/Prod:output:0)model_5/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:ť
#model_5/dense_1/Tensordot/transpose	Transposemodel_5/concat/concat:output:0)model_5/dense_1/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ş
!model_5/dense_1/Tensordot/ReshapeReshape'model_5/dense_1/Tensordot/transpose:y:0(model_5/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ť
 model_5/dense_1/Tensordot/MatMulMatMul*model_5/dense_1/Tensordot/Reshape:output:00model_5/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙×8l
!model_5/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:×8i
'model_5/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ç
"model_5/dense_1/Tensordot/concat_1ConcatV2+model_5/dense_1/Tensordot/GatherV2:output:0*model_5/dense_1/Tensordot/Const_2:output:00model_5/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:˝
model_5/dense_1/TensordotReshape*model_5/dense_1/Tensordot/MatMul:product:0+model_5/dense_1/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8
&model_5/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_5_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:×8*
dtype0ś
model_5/dense_1/BiasAddBiasAdd"model_5/dense_1/Tensordot:output:0.model_5/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8
model_5/dense_1/SoftmaxSoftmax model_5/dense_1/BiasAdd:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8~
IdentityIdentity!model_5/dense_1/Softmax:softmax:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8o

Identity_1Identitymodel_5/lstm_3/while:output:4^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o

Identity_2Identitymodel_5/lstm_3/while:output:5^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp0^model_5/attention_layer/MatMul_1/ReadVariableOp3^model_5/attention_layer/transpose_1/ReadVariableOp3^model_5/attention_layer/transpose_2/ReadVariableOp^model_5/attention_layer/while'^model_5/dense_1/BiasAdd/ReadVariableOp)^model_5/dense_1/Tensordot/ReadVariableOp%^model_5/embedding_1/embedding_lookup(^model_5/lstm_3/lstm_cell/ReadVariableOp*^model_5/lstm_3/lstm_cell/ReadVariableOp_1*^model_5/lstm_3/lstm_cell/ReadVariableOp_2*^model_5/lstm_3/lstm_cell/ReadVariableOp_3.^model_5/lstm_3/lstm_cell/split/ReadVariableOp0^model_5/lstm_3/lstm_cell/split_1/ReadVariableOp^model_5/lstm_3/while*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : 2b
/model_5/attention_layer/MatMul_1/ReadVariableOp/model_5/attention_layer/MatMul_1/ReadVariableOp2h
2model_5/attention_layer/transpose_1/ReadVariableOp2model_5/attention_layer/transpose_1/ReadVariableOp2h
2model_5/attention_layer/transpose_2/ReadVariableOp2model_5/attention_layer/transpose_2/ReadVariableOp2>
model_5/attention_layer/whilemodel_5/attention_layer/while2P
&model_5/dense_1/BiasAdd/ReadVariableOp&model_5/dense_1/BiasAdd/ReadVariableOp2T
(model_5/dense_1/Tensordot/ReadVariableOp(model_5/dense_1/Tensordot/ReadVariableOp2L
$model_5/embedding_1/embedding_lookup$model_5/embedding_1/embedding_lookup2R
'model_5/lstm_3/lstm_cell/ReadVariableOp'model_5/lstm_3/lstm_cell/ReadVariableOp2V
)model_5/lstm_3/lstm_cell/ReadVariableOp_1)model_5/lstm_3/lstm_cell/ReadVariableOp_12V
)model_5/lstm_3/lstm_cell/ReadVariableOp_2)model_5/lstm_3/lstm_cell/ReadVariableOp_22V
)model_5/lstm_3/lstm_cell/ReadVariableOp_3)model_5/lstm_3/lstm_cell/ReadVariableOp_32^
-model_5/lstm_3/lstm_cell/split/ReadVariableOp-model_5/lstm_3/lstm_cell/split/ReadVariableOp2b
/model_5/lstm_3/lstm_cell/split_1/ReadVariableOp/model_5/lstm_3/lstm_cell/split_1/ReadVariableOp2,
model_5/lstm_3/whilemodel_5/lstm_3/while:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2:VR
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_6:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_7:%!

_user_specified_name51638:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
n
ô
(model_5_attention_layer_while_body_51934L
Hmodel_5_attention_layer_while_model_5_attention_layer_while_loop_counterR
Nmodel_5_attention_layer_while_model_5_attention_layer_while_maximum_iterations-
)model_5_attention_layer_while_placeholder/
+model_5_attention_layer_while_placeholder_1/
+model_5_attention_layer_while_placeholder_2I
Emodel_5_attention_layer_while_model_5_attention_layer_strided_slice_0
model_5_attention_layer_while_tensorarrayv2read_tensorlistgetitem_model_5_attention_layer_tensorarrayunstack_tensorlistfromtensor_01
-model_5_attention_layer_while_shape_input_8_0S
?model_5_attention_layer_while_shape_1_readvariableop_resource_0:
T
@model_5_attention_layer_while_matmul_1_readvariableop_resource_0:
R
?model_5_attention_layer_while_shape_3_readvariableop_resource_0:	*
&model_5_attention_layer_while_identity,
(model_5_attention_layer_while_identity_1,
(model_5_attention_layer_while_identity_2,
(model_5_attention_layer_while_identity_3,
(model_5_attention_layer_while_identity_4G
Cmodel_5_attention_layer_while_model_5_attention_layer_strided_slice
model_5_attention_layer_while_tensorarrayv2read_tensorlistgetitem_model_5_attention_layer_tensorarrayunstack_tensorlistfromtensor/
+model_5_attention_layer_while_shape_input_8Q
=model_5_attention_layer_while_shape_1_readvariableop_resource:
R
>model_5_attention_layer_while_matmul_1_readvariableop_resource:
P
=model_5_attention_layer_while_shape_3_readvariableop_resource:	˘5model_5/attention_layer/while/MatMul_1/ReadVariableOp˘6model_5/attention_layer/while/transpose/ReadVariableOp˘8model_5/attention_layer/while/transpose_1/ReadVariableOp 
Omodel_5/attention_layer/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    
Amodel_5/attention_layer/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmodel_5_attention_layer_while_tensorarrayv2read_tensorlistgetitem_model_5_attention_layer_tensorarrayunstack_tensorlistfromtensor_0)model_5_attention_layer_while_placeholderXmodel_5/attention_layer/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0
#model_5/attention_layer/while/ShapeShape-model_5_attention_layer_while_shape_input_8_0*
T0*
_output_shapes
::íĎ
%model_5/attention_layer/while/unstackUnpack,model_5/attention_layer/while/Shape:output:0*
T0*
_output_shapes
: : : *	
numś
4model_5/attention_layer/while/Shape_1/ReadVariableOpReadVariableOp?model_5_attention_layer_while_shape_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%model_5/attention_layer/while/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      
'model_5/attention_layer/while/unstack_1Unpack.model_5/attention_layer/while/Shape_1:output:0*
T0*
_output_shapes
: : *	
num|
+model_5/attention_layer/while/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Č
%model_5/attention_layer/while/ReshapeReshape-model_5_attention_layer_while_shape_input_8_04model_5/attention_layer/while/Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
6model_5/attention_layer/while/transpose/ReadVariableOpReadVariableOp?model_5_attention_layer_while_shape_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0}
,model_5/attention_layer/while/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       Ö
'model_5/attention_layer/while/transpose	Transpose>model_5/attention_layer/while/transpose/ReadVariableOp:value:05model_5/attention_layer/while/transpose/perm:output:0*
T0* 
_output_shapes
:
~
-model_5/attention_layer/while/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙Â
'model_5/attention_layer/while/Reshape_1Reshape+model_5/attention_layer/while/transpose:y:06model_5/attention_layer/while/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
Ă
$model_5/attention_layer/while/MatMulMatMul.model_5/attention_layer/while/Reshape:output:00model_5/attention_layer/while/Reshape_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
/model_5/attention_layer/while/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :r
/model_5/attention_layer/while/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
-model_5/attention_layer/while/Reshape_2/shapePack.model_5/attention_layer/while/unstack:output:08model_5/attention_layer/while/Reshape_2/shape/1:output:08model_5/attention_layer/while/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:Ň
'model_5/attention_layer/while/Reshape_2Reshape.model_5/attention_layer/while/MatMul:product:06model_5/attention_layer/while/Reshape_2/shape:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙¸
5model_5/attention_layer/while/MatMul_1/ReadVariableOpReadVariableOp@model_5_attention_layer_while_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0ě
&model_5/attention_layer/while/MatMul_1MatMulHmodel_5/attention_layer/while/TensorArrayV2Read/TensorListGetItem:item:0=model_5/attention_layer/while/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
,model_5/attention_layer/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Ö
(model_5/attention_layer/while/ExpandDims
ExpandDims0model_5/attention_layer/while/MatMul_1:product:05model_5/attention_layer/while/ExpandDims/dim:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙Ç
!model_5/attention_layer/while/addAddV20model_5/attention_layer/while/Reshape_2:output:01model_5/attention_layer/while/ExpandDims:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
"model_5/attention_layer/while/TanhTanh%model_5/attention_layer/while/add:z:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
%model_5/attention_layer/while/Shape_2Shape&model_5/attention_layer/while/Tanh:y:0*
T0*
_output_shapes
::íĎ
'model_5/attention_layer/while/unstack_2Unpack.model_5/attention_layer/while/Shape_2:output:0*
T0*
_output_shapes
: : : *	
numľ
4model_5/attention_layer/while/Shape_3/ReadVariableOpReadVariableOp?model_5_attention_layer_while_shape_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0v
%model_5/attention_layer/while/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      
'model_5/attention_layer/while/unstack_3Unpack.model_5/attention_layer/while/Shape_3:output:0*
T0*
_output_shapes
: : *	
num~
-model_5/attention_layer/while/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ĺ
'model_5/attention_layer/while/Reshape_3Reshape&model_5/attention_layer/while/Tanh:y:06model_5/attention_layer/while/Reshape_3/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙š
8model_5/attention_layer/while/transpose_1/ReadVariableOpReadVariableOp?model_5_attention_layer_while_shape_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0
.model_5/attention_layer/while/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Ű
)model_5/attention_layer/while/transpose_1	Transpose@model_5/attention_layer/while/transpose_1/ReadVariableOp:value:07model_5/attention_layer/while/transpose_1/perm:output:0*
T0*
_output_shapes
:	~
-model_5/attention_layer/while/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙Ă
'model_5/attention_layer/while/Reshape_4Reshape-model_5/attention_layer/while/transpose_1:y:06model_5/attention_layer/while/Reshape_4/shape:output:0*
T0*
_output_shapes
:	Ć
&model_5/attention_layer/while/MatMul_2MatMul0model_5/attention_layer/while/Reshape_3:output:00model_5/attention_layer/while/Reshape_4:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
/model_5/attention_layer/while/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value
B :q
/model_5/attention_layer/while/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
-model_5/attention_layer/while/Reshape_5/shapePack0model_5/attention_layer/while/unstack_2:output:08model_5/attention_layer/while/Reshape_5/shape/1:output:08model_5/attention_layer/while/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:Ó
'model_5/attention_layer/while/Reshape_5Reshape0model_5/attention_layer/while/MatMul_2:product:06model_5/attention_layer/while/Reshape_5/shape:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
%model_5/attention_layer/while/SqueezeSqueeze0model_5/attention_layer/while/Reshape_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims

˙˙˙˙˙˙˙˙˙
%model_5/attention_layer/while/SoftmaxSoftmax.model_5/attention_layer/while/Squeeze:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Bmodel_5/attention_layer/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem+model_5_attention_layer_while_placeholder_1)model_5_attention_layer_while_placeholder/model_5/attention_layer/while/Softmax:softmax:0*
_output_shapes
: *
element_dtype0:éčŇg
%model_5/attention_layer/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¨
#model_5/attention_layer/while/add_1AddV2)model_5_attention_layer_while_placeholder.model_5/attention_layer/while/add_1/y:output:0*
T0*
_output_shapes
: g
%model_5/attention_layer/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :Ç
#model_5/attention_layer/while/add_2AddV2Hmodel_5_attention_layer_while_model_5_attention_layer_while_loop_counter.model_5/attention_layer/while/add_2/y:output:0*
T0*
_output_shapes
: Ą
&model_5/attention_layer/while/IdentityIdentity'model_5/attention_layer/while/add_2:z:0#^model_5/attention_layer/while/NoOp*
T0*
_output_shapes
: Ę
(model_5/attention_layer/while/Identity_1IdentityNmodel_5_attention_layer_while_model_5_attention_layer_while_maximum_iterations#^model_5/attention_layer/while/NoOp*
T0*
_output_shapes
: Ł
(model_5/attention_layer/while/Identity_2Identity'model_5/attention_layer/while/add_1:z:0#^model_5/attention_layer/while/NoOp*
T0*
_output_shapes
: Î
(model_5/attention_layer/while/Identity_3IdentityRmodel_5/attention_layer/while/TensorArrayV2Write/TensorListSetItem:output_handle:0#^model_5/attention_layer/while/NoOp*
T0*
_output_shapes
: ˝
(model_5/attention_layer/while/Identity_4Identity/model_5/attention_layer/while/Softmax:softmax:0#^model_5/attention_layer/while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ě
"model_5/attention_layer/while/NoOpNoOp6^model_5/attention_layer/while/MatMul_1/ReadVariableOp7^model_5/attention_layer/while/transpose/ReadVariableOp9^model_5/attention_layer/while/transpose_1/ReadVariableOp*
_output_shapes
 "Y
&model_5_attention_layer_while_identity/model_5/attention_layer/while/Identity:output:0"]
(model_5_attention_layer_while_identity_11model_5/attention_layer/while/Identity_1:output:0"]
(model_5_attention_layer_while_identity_21model_5/attention_layer/while/Identity_2:output:0"]
(model_5_attention_layer_while_identity_31model_5/attention_layer/while/Identity_3:output:0"]
(model_5_attention_layer_while_identity_41model_5/attention_layer/while/Identity_4:output:0"
>model_5_attention_layer_while_matmul_1_readvariableop_resource@model_5_attention_layer_while_matmul_1_readvariableop_resource_0"
Cmodel_5_attention_layer_while_model_5_attention_layer_strided_sliceEmodel_5_attention_layer_while_model_5_attention_layer_strided_slice_0"
=model_5_attention_layer_while_shape_1_readvariableop_resource?model_5_attention_layer_while_shape_1_readvariableop_resource_0"
=model_5_attention_layer_while_shape_3_readvariableop_resource?model_5_attention_layer_while_shape_3_readvariableop_resource_0"\
+model_5_attention_layer_while_shape_input_8-model_5_attention_layer_while_shape_input_8_0"
model_5_attention_layer_while_tensorarrayv2read_tensorlistgetitem_model_5_attention_layer_tensorarrayunstack_tensorlistfromtensormodel_5_attention_layer_while_tensorarrayv2read_tensorlistgetitem_model_5_attention_layer_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙: : : 2n
5model_5/attention_layer/while/MatMul_1/ReadVariableOp5model_5/attention_layer/while/MatMul_1/ReadVariableOp2p
6model_5/attention_layer/while/transpose/ReadVariableOp6model_5/attention_layer/while/transpose/ReadVariableOp2t
8model_5/attention_layer/while/transpose_1/ReadVariableOp8model_5/attention_layer/while/transpose_1/ReadVariableOp:b ^

_output_shapes
: 
D
_user_specified_name,*model_5/attention_layer/while/loop_counter:hd

_output_shapes
: 
J
_user_specified_name20model_5/attention_layer/while/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:]Y

_output_shapes
: 
?
_user_specified_name'%model_5/attention_layer/strided_slice:ws

_output_shapes
: 
Y
_user_specified_nameA?model_5/attention_layer/TensorArrayUnstack/TensorListFromTensor:VR
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource

˙
&__inference_lstm_3_layer_call_fn_53857

inputs
initial_state_0
initial_state_1
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1

identity_2˘StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsinitial_state_0initial_state_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_53039}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:YU
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinitial_state_0:YU
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinitial_state_1:%!

_user_specified_name53845:%!

_user_specified_name53847:%!

_user_specified_name53849
Ĺ
ô
)__inference_lstm_cell_layer_call_fn_55425

inputs
states_0
states_1
unknown:

	unknown_0:	
	unknown_1:

identity

identity_1

identity_2˘StatefulPartitionedCallŞ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_52519p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_0:RN
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
states_1:%!

_user_specified_name55413:%!

_user_specified_name55415:%!

_user_specified_name55417
Á	
ž
while_cond_54042
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_54042___redundant_placeholder03
/while_while_cond_54042___redundant_placeholder13
/while_while_cond_54042___redundant_placeholder23
/while_while_cond_54042___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:

_output_shapes
:
ť	
ź
while_cond_54633
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice3
/while_while_cond_54633___redundant_placeholder03
/while_while_cond_54633___redundant_placeholder13
/while_while_cond_54633___redundant_placeholder23
/while_while_cond_54633___redundant_placeholder3
while_identity
`

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: :::::J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:

_output_shapes
:
Ś

'__inference_dense_1_layer_call_fn_55360

inputs
unknown:
×8
	unknown_0:	×8
identity˘StatefulPartitionedCallč
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_53363}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:%!

_user_specified_name55354:%!

_user_specified_name55356
Ť
¤
F__inference_embedding_1_layer_call_and_return_conditional_losses_53810

inputs*
embedding_lookup_53805:
×8
identity˘embedding_lookup^
CastCastinputs*

DstT0*

SrcT0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ă
embedding_lookupResourceGatherembedding_lookup_53805Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/53805*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
dtype0
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: 2$
embedding_lookupembedding_lookup:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:%!

_user_specified_name53805
Ľt

A__inference_lstm_3_layer_call_and_return_conditional_losses_53606

inputs
initial_state
initial_state_1;
'lstm_cell_split_readvariableop_resource:
8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity

identity_1

identity_2˘lstm_cell/ReadVariableOp˘lstm_cell/ReadVariableOp_1˘lstm_cell/ReadVariableOp_2˘lstm_cell/ReadVariableOp_3˘lstm_cell/split/ReadVariableOp˘ lstm_cell/split_1/ReadVariableOp˘whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::íĎ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskl
lstm_cell/ones_likeOnesLikestrided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
lstm_cell/ones_like_1OnesLikeinitial_state*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell/mulMulstrided_slice_1:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/mul_1Mulstrided_slice_1:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/mul_2Mulstrided_slice_1:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/mul_3Mulstrided_slice_1:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype0Ä
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitz
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ś
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
lstm_cell/mul_4Mulinitial_statelstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
lstm_cell/mul_5Mulinitial_statelstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
lstm_cell/mul_6Mulinitial_statelstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
lstm_cell/mul_7Mulinitial_statelstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0initial_state_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Đ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_stateinitial_state_1strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_53474*
condR
while_cond_53473*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕ
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:WS
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameinitial_state:WS
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nameinitial_state:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
é
˛
'__inference_model_5_layer_call_fn_53663
input_2
input_8
input_6
input_7
unknown:
×8
	unknown_0:

	unknown_1:	
	unknown_2:

	unknown_3:

	unknown_4:

	unknown_5:	
	unknown_6:
×8
	unknown_7:	×8
identity

identity_1

identity_2˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2input_8input_6input_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*+
_read_only_resource_inputs
		
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_5_layer_call_and_return_conditional_losses_53372}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2:VR
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_6:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_7:%!

_user_specified_name53639:%!

_user_specified_name53641:%!

_user_specified_name53643:%!

_user_specified_name53645:%!

_user_specified_name53647:%	!

_user_specified_name53649:%
!

_user_specified_name53651:%!

_user_specified_name53653:%!

_user_specified_name53655
ío
	
while_body_54344
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
˘while/lstm_cell/ReadVariableOp˘ while/lstm_cell/ReadVariableOp_1˘ while/lstm_cell/ReadVariableOp_2˘ while/lstm_cell/ReadVariableOp_3˘$while/lstm_cell/split/ReadVariableOp˘&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0
while/lstm_cell/ones_likeOnesLike0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
while/lstm_cell/ones_like_1OnesLikewhile_placeholder_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ö
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Č
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_4Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_5Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_6Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_7Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ˝
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:GC

_output_shapes
: 
)
_user_specified_namestrided_slice_1:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource

k
A__inference_concat_layer_call_and_return_conditional_losses_53331

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙e
IdentityIdentityconcat:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:]Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
őy

J__inference_attention_layer_layer_call_and_return_conditional_losses_55338
inputs_0
inputs_13
shape_2_readvariableop_resource:
4
 matmul_1_readvariableop_resource:
2
shape_4_readvariableop_resource:	
identity

identity_1˘MatMul_1/ReadVariableOp˘transpose_1/ReadVariableOp˘transpose_2/ReadVariableOp˘whileW
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :g
SumSuminputs_0Sum/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :k
Sum_1Suminputs_0 Sum_1/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_1transpose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::íĎ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskM
Shape_1Shapeinputs_0*
T0*
_output_shapes
::íĎS
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
numx
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource* 
_output_shapes
:
*
dtype0X
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"      S
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   g
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
transpose_1/ReadVariableOpReadVariableOpshape_2_readvariableop_resource* 
_output_shapes
:
*
dtype0a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_1	Transpose"transpose_1/ReadVariableOp:value:0transpose_1/perm:output:0*
T0* 
_output_shapes
:
`
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙j
	Reshape_1Reshapetranspose_1:y:0Reshape_1/shape:output:0*
T0* 
_output_shapes
:
i
MatMulMatMulReshape:output:0Reshape_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙T
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :T
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:x
	Reshape_2ReshapeMatMul:product:0Reshape_2/shape:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype0
MatMul_1MatMulstrided_slice_1:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :|

ExpandDims
ExpandDimsMatMul_1:product:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙m
addAddV2Reshape_2:output:0ExpandDims:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙M
TanhTanhadd:z:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙M
Shape_3ShapeTanh:y:0*
T0*
_output_shapes
::íĎU
	unstack_2UnpackShape_3:output:0*
T0*
_output_shapes
: : : *	
numw
Shape_4/ReadVariableOpReadVariableOpshape_4_readvariableop_resource*
_output_shapes
:	*
dtype0X
Shape_4Const*
_output_shapes
:*
dtype0*
valueB"      S
	unstack_3UnpackShape_4:output:0*
T0*
_output_shapes
: : *	
num`
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   k
	Reshape_3ReshapeTanh:y:0Reshape_3/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙{
transpose_2/ReadVariableOpReadVariableOpshape_4_readvariableop_resource*
_output_shapes
:	*
dtype0a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       
transpose_2	Transpose"transpose_2/ReadVariableOp:value:0transpose_2/perm:output:0*
T0*
_output_shapes
:	`
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙i
	Reshape_4Reshapetranspose_2:y:0Reshape_4/shape:output:0*
T0*
_output_shapes
:	l
MatMul_2MatMulReshape_3:output:0Reshape_4:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙T
Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value
B :S
Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :
Reshape_5/shapePackunstack_2:output:0Reshape_5/shape/1:output:0Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:y
	Reshape_5ReshapeMatMul_2:product:0Reshape_5/shape:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙y
SqueezeSqueezeReshape_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims

˙˙˙˙˙˙˙˙˙W
SoftmaxSoftmaxSqueeze:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Á
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0Sum_1:output:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0inputs_0shape_2_readvariableop_resource matmul_1_readvariableop_resourceshape_4_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*S
_output_shapesA
?: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙: : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_55148*
condR
while_cond_55147*R
output_shapesA
?: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙: : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_3	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_3/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙e
transpose_4/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_4	Transposetranspose_3:y:0transpose_4/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙T
Shape_5Shapetranspose_4:y:0*
T0*
_output_shapes
::íĎ_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ű
strided_slice_3StridedSliceShape_5:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
TensorArrayV2_3/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙¸
TensorArrayV2_3TensorListReserve&TensorArrayV2_3/element_shape:output:0strided_slice_3:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
7TensorArrayUnstack_1/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙  ć
)TensorArrayUnstack_1/TensorListFromTensorTensorListFromTensortranspose_4:y:0@TensorArrayUnstack_1/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
strided_slice_4StridedSlicetranspose_4:y:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_mask[
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
ExpandDims_1
ExpandDimsstrided_slice_4:output:0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙c
mulMulinputs_0ExpandDims_1:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :j
Sum_2Summul:z:0 Sum_2/reduction_indices:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_4/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ¸
TensorArrayV2_4TensorListReserve&TensorArrayV2_4/element_shape:output:0strided_slice_3:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇH
time_1Const*
_output_shapes
: *
dtype0*
value	B : e
while_1/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙V
while_1/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ŕ
while_1StatelessWhilewhile_1/loop_counter:output:0#while_1/maximum_iterations:output:0time_1:output:0TensorArrayV2_4:handle:0Sum:output:0strided_slice_3:output:09TensorArrayUnstack_1/TensorListFromTensor:output_handle:0inputs_0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*M
_output_shapes;
9: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *
bodyR
while_1_body_55280*
condR
while_1_cond_55279*L
output_shapes;
9: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙*
parallel_iterations 
2TensorArrayV2Stack_1/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ň
$TensorArrayV2Stack_1/TensorListStackTensorListStackwhile_1:output:3;TensorArrayV2Stack_1/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_5StridedSlice-TensorArrayV2Stack_1/TensorListStack:tensor:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_5/permConst*
_output_shapes
:*
dtype0*!
valueB"          ˘
transpose_5	Transpose-TensorArrayV2Stack_1/TensorListStack:tensor:0transpose_5/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙l
IdentityIdentitytranspose_5:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙n

Identity_1Identitytranspose_3:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙~
NoOpNoOp^MatMul_1/ReadVariableOp^transpose_1/ReadVariableOp^transpose_2/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp28
transpose_1/ReadVariableOptranspose_1/ReadVariableOp28
transpose_2/ReadVariableOptranspose_2/ReadVariableOp2
whilewhile:W S
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0:_[
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ˇt

A__inference_lstm_3_layer_call_and_return_conditional_losses_55056

inputs
initial_state_0
initial_state_1;
'lstm_cell_split_readvariableop_resource:
8
)lstm_cell_split_1_readvariableop_resource:	5
!lstm_cell_readvariableop_resource:

identity

identity_1

identity_2˘lstm_cell/ReadVariableOp˘lstm_cell/ReadVariableOp_1˘lstm_cell/ReadVariableOp_2˘lstm_cell/ReadVariableOp_3˘lstm_cell/split/ReadVariableOp˘ lstm_cell/split_1/ReadVariableOp˘whilec
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙P
ShapeShapetranspose:y:0*
T0*
_output_shapes
::íĎ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ń
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙˛
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ŕ
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ę
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maskl
lstm_cell/ones_likeOnesLikestrided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙e
lstm_cell/ones_like_1OnesLikeinitial_state_0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙z
lstm_cell/mulMulstrided_slice_1:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/mul_1Mulstrided_slice_1:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/mul_2Mulstrided_slice_1:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/mul_3Mulstrided_slice_1:output:0lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙[
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
*
dtype0Ä
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_splitz
lstm_cell/MatMulMatMullstm_cell/mul:z:0lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_1MatMullstm_cell/mul_1:z:0lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_2MatMullstm_cell/mul_2:z:0lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/MatMul_3MatMullstm_cell/mul_3:z:0lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙]
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
:*
dtype0ś
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
lstm_cell/mul_4Mulinitial_state_0lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
lstm_cell/mul_5Mulinitial_state_0lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
lstm_cell/mul_6Mulinitial_state_0lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
lstm_cell/mul_7Mulinitial_state_0lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙|
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0n
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        p
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       p
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_4MatMullstm_cell/mul_4:z:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙b
lstm_cell/SigmoidSigmoidlstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_5MatMullstm_cell/mul_5:z:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_1AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell/Sigmoid_1Sigmoidlstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙s
lstm_cell/mul_8Mullstm_cell/Sigmoid_1:y:0initial_state_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_6MatMullstm_cell/mul_6:z:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_2AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙^
lstm_cell/TanhTanhlstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lstm_cell/mul_9Mullstm_cell/Sigmoid:y:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙u
lstm_cell/add_3AddV2lstm_cell/mul_8:z:0lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙~
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
*
dtype0p
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       r
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        r
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Š
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
lstm_cell/MatMul_7MatMullstm_cell/mul_7:z:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
lstm_cell/add_4AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙f
lstm_cell/Sigmoid_2Sigmoidlstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`
lstm_cell/Tanh_1Tanhlstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y
lstm_cell/mul_10Mullstm_cell/Sigmoid_2:y:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   ś
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éčŇF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ň
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0initial_state_0initial_state_1strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_54924*
condR
while_cond_54923*M
output_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   Ě
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
element_dtype0h
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    l
IdentityIdentitytranspose_1:y:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙`

Identity_1Identitywhile:output:4^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙`

Identity_2Identitywhile:output:5^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ŕ
NoOpNoOp^lstm_cell/ReadVariableOp^lstm_cell/ReadVariableOp_1^lstm_cell/ReadVariableOp_2^lstm_cell/ReadVariableOp_3^lstm_cell/split/ReadVariableOp!^lstm_cell/split_1/ReadVariableOp^while*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : 24
lstm_cell/ReadVariableOplstm_cell/ReadVariableOp28
lstm_cell/ReadVariableOp_1lstm_cell/ReadVariableOp_128
lstm_cell/ReadVariableOp_2lstm_cell/ReadVariableOp_228
lstm_cell/ReadVariableOp_3lstm_cell/ReadVariableOp_32@
lstm_cell/split/ReadVariableOplstm_cell/split/ReadVariableOp2D
 lstm_cell/split_1/ReadVariableOp lstm_cell/split_1/ReadVariableOp2
whilewhile:] Y
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:YU
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinitial_state_0:YU
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinitial_state_1:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ăo
˙
while_body_53474
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0C
/while_lstm_cell_split_readvariableop_resource_0:
@
1while_lstm_cell_split_1_readvariableop_resource_0:	=
)while_lstm_cell_readvariableop_resource_0:

while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorA
-while_lstm_cell_split_readvariableop_resource:
>
/while_lstm_cell_split_1_readvariableop_resource:	;
'while_lstm_cell_readvariableop_resource:
˘while/lstm_cell/ReadVariableOp˘ while/lstm_cell/ReadVariableOp_1˘ while/lstm_cell/ReadVariableOp_2˘ while/lstm_cell/ReadVariableOp_3˘$while/lstm_cell/split/ReadVariableOp˘&while/lstm_cell/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0
while/lstm_cell/ones_likeOnesLike0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙o
while/lstm_cell/ones_like_1OnesLikewhile_placeholder_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
while/lstm_cell/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/ones_like:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙a
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
*
dtype0Ö
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
:
:
:
*
	num_split
while/lstm_cell/MatMulMatMulwhile/lstm_cell/mul:z:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_1MatMulwhile/lstm_cell/mul_1:z:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_2MatMulwhile/lstm_cell/mul_2:z:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/MatMul_3MatMulwhile/lstm_cell/mul_3:z:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙c
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
:*
dtype0Č
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
::::*
	num_split
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_4Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_5Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_6Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_7Mulwhile_placeholder_2while/lstm_cell/ones_like_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0t
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        v
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       v
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ˝
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_4MatMulwhile/lstm_cell/mul_4:z:0&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙n
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_5MatMulwhile/lstm_cell/mul_5:z:0(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_1AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_8Mulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_6MatMulwhile/lstm_cell/mul_6:z:0(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙j
while/lstm_cell/TanhTanhwhile/lstm_cell/add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_9Mulwhile/lstm_cell/Sigmoid:y:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_3AddV2while/lstm_cell/mul_8:z:0while/lstm_cell/mul_9:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
*
dtype0v
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       x
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        x
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ç
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
*

begin_mask*
end_mask
while/lstm_cell/MatMul_7MatMulwhile/lstm_cell/mul_7:z:0(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙r
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙l
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/lstm_cell/mul_10Mulwhile/lstm_cell/Sigmoid_2:y:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ă
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_10:z:0*
_output_shapes
: *
element_dtype0:éčŇM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell/mul_10:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙w
while/Identity_5Identitywhile/lstm_cell/add_3:z:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

while/NoOpNoOp^while/lstm_cell/ReadVariableOp!^while/lstm_cell/ReadVariableOp_1!^while/lstm_cell/ReadVariableOp_2!^while/lstm_cell/ReadVariableOp_3%^while/lstm_cell/split/ReadVariableOp'^while/lstm_cell/split_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : 2@
while/lstm_cell/ReadVariableOpwhile/lstm_cell/ReadVariableOp2D
 while/lstm_cell/ReadVariableOp_1 while/lstm_cell/ReadVariableOp_12D
 while/lstm_cell/ReadVariableOp_2 while/lstm_cell/ReadVariableOp_22D
 while/lstm_cell/ReadVariableOp_3 while/lstm_cell/ReadVariableOp_32L
$while/lstm_cell/split/ReadVariableOp$while/lstm_cell/split/ReadVariableOp2P
&while/lstm_cell/split_1/ReadVariableOp&while/lstm_cell/split_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource
ö
â
/__inference_attention_layer_layer_call_fn_55070
inputs_0
inputs_1
unknown:

	unknown_0:

	unknown_1:	
identity

identity_1˘StatefulPartitionedCallŹ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *V
_output_shapesD
B:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_attention_layer_layer_call_and_return_conditional_losses_53316}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_0:_[
5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
"
_user_specified_name
inputs_1:%!

_user_specified_name55060:%!

_user_specified_name55062:%!

_user_specified_name55064
Ę$

B__inference_model_5_layer_call_and_return_conditional_losses_53633
input_2
input_8
input_6
input_7%
embedding_1_53378:
×8 
lstm_3_53607:

lstm_3_53609:	 
lstm_3_53611:
)
attention_layer_53616:
)
attention_layer_53618:
(
attention_layer_53620:	!
dense_1_53625:
×8
dense_1_53627:	×8
identity

identity_1

identity_2˘'attention_layer/StatefulPartitionedCall˘dense_1/StatefulPartitionedCall˘#embedding_1/StatefulPartitionedCall˘lstm_3/StatefulPartitionedCallö
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_53378*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_embedding_1_layer_call_and_return_conditional_losses_52682ę
lstm_3/StatefulPartitionedCallStatefulPartitionedCall,embedding_1/StatefulPartitionedCall:output:0input_6input_7lstm_3_53607lstm_3_53609lstm_3_53611*
Tin

2*
Tout
2*
_collective_manager_ids
 *]
_output_shapesK
I:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_lstm_3_layer_call_and_return_conditional_losses_53606
'attention_layer/StatefulPartitionedCallStatefulPartitionedCallinput_8'lstm_3/StatefulPartitionedCall:output:0attention_layer_53616attention_layer_53618attention_layer_53620*
Tin	
2*
Tout
2*
_collective_manager_ids
 *V
_output_shapesD
B:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_attention_layer_layer_call_and_return_conditional_losses_53316
concat/PartitionedCallPartitionedCall'lstm_3/StatefulPartitionedCall:output:00attention_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_concat_layer_call_and_return_conditional_losses_53331
dense_1/StatefulPartitionedCallStatefulPartitionedCallconcat/PartitionedCall:output:0dense_1_53625dense_1_53627*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_53363
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8y

Identity_1Identity'lstm_3/StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙y

Identity_2Identity'lstm_3/StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙ľ
NoOpNoOp(^attention_layer/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall^lstm_3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*
_input_shapesq
o:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : : : : 2R
'attention_layer/StatefulPartitionedCall'attention_layer/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2@
lstm_3/StatefulPartitionedCalllstm_3/StatefulPartitionedCall:Y U
0
_output_shapes
:˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_2:VR
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_8:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_6:QM
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
!
_user_specified_name	input_7:%!

_user_specified_name53378:%!

_user_specified_name53607:%!

_user_specified_name53609:%!

_user_specified_name53611:%!

_user_specified_name53616:%	!

_user_specified_name53618:%
!

_user_specified_name53620:%!

_user_specified_name53625:%!

_user_specified_name53627
ńR
	
__inference__traced_save_55726
file_prefixA
-read_disablecopyonread_embedding_1_embeddings:
×8@
,read_1_disablecopyonread_attention_layer_w_a:
@
,read_2_disablecopyonread_attention_layer_u_a:
?
,read_3_disablecopyonread_attention_layer_v_a:	;
'read_4_disablecopyonread_dense_1_kernel:
×84
%read_5_disablecopyonread_dense_1_bias:	×8D
0read_6_disablecopyonread_lstm_3_lstm_cell_kernel:
N
:read_7_disablecopyonread_lstm_3_lstm_cell_recurrent_kernel:
=
.read_8_disablecopyonread_lstm_3_lstm_cell_bias:	
savev2_const
identity_19˘MergeV2Checkpoints˘Read/DisableCopyOnRead˘Read/ReadVariableOp˘Read_1/DisableCopyOnRead˘Read_1/ReadVariableOp˘Read_2/DisableCopyOnRead˘Read_2/ReadVariableOp˘Read_3/DisableCopyOnRead˘Read_3/ReadVariableOp˘Read_4/DisableCopyOnRead˘Read_4/ReadVariableOp˘Read_5/DisableCopyOnRead˘Read_5/ReadVariableOp˘Read_6/DisableCopyOnRead˘Read_6/ReadVariableOp˘Read_7/DisableCopyOnRead˘Read_7/ReadVariableOp˘Read_8/DisableCopyOnRead˘Read_8/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
Read/DisableCopyOnReadDisableCopyOnRead-read_disablecopyonread_embedding_1_embeddings"/device:CPU:0*
_output_shapes
 Ť
Read/ReadVariableOpReadVariableOp-read_disablecopyonread_embedding_1_embeddings^Read/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
×8*
dtype0k
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
×8c

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0* 
_output_shapes
:
×8
Read_1/DisableCopyOnReadDisableCopyOnRead,read_1_disablecopyonread_attention_layer_w_a"/device:CPU:0*
_output_shapes
 Ž
Read_1/ReadVariableOpReadVariableOp,read_1_disablecopyonread_attention_layer_w_a^Read_1/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0o

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
e

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_2/DisableCopyOnReadDisableCopyOnRead,read_2_disablecopyonread_attention_layer_u_a"/device:CPU:0*
_output_shapes
 Ž
Read_2/ReadVariableOpReadVariableOp,read_2_disablecopyonread_attention_layer_u_a^Read_2/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0o

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
e

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_3/DisableCopyOnReadDisableCopyOnRead,read_3_disablecopyonread_attention_layer_v_a"/device:CPU:0*
_output_shapes
 ­
Read_3/ReadVariableOpReadVariableOp,read_3_disablecopyonread_attention_layer_v_a^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	*
dtype0n

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	d

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:	{
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 Š
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_1_kernel^Read_4/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
×8*
dtype0o

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
×8e

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0* 
_output_shapes
:
×8y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 ˘
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_1_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:×8*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:×8b
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:×8
Read_6/DisableCopyOnReadDisableCopyOnRead0read_6_disablecopyonread_lstm_3_lstm_cell_kernel"/device:CPU:0*
_output_shapes
 ˛
Read_6/ReadVariableOpReadVariableOp0read_6_disablecopyonread_lstm_3_lstm_cell_kernel^Read_6/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0p
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_7/DisableCopyOnReadDisableCopyOnRead:read_7_disablecopyonread_lstm_3_lstm_cell_recurrent_kernel"/device:CPU:0*
_output_shapes
 ź
Read_7/ReadVariableOpReadVariableOp:read_7_disablecopyonread_lstm_3_lstm_cell_recurrent_kernel^Read_7/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
*
dtype0p
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
g
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0* 
_output_shapes
:

Read_8/DisableCopyOnReadDisableCopyOnRead.read_8_disablecopyonread_lstm_3_lstm_cell_bias"/device:CPU:0*
_output_shapes
 Ť
Read_8/ReadVariableOpReadVariableOp.read_8_disablecopyonread_lstm_3_lstm_cell_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:b
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ë
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*ô
valueęBç
B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/W_a/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/U_a/.ATTRIBUTES/VARIABLE_VALUEB3layer_with_weights-2/V_a/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:
*
dtype0*'
valueB
B B B B B B B B B B ˘
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2

&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:ł
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_18Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_19IdentityIdentity_18:output:0^NoOp*
T0*
_output_shapes
: ţ
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp*
_output_shapes
 "#
identity_19Identity_19:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:62
0
_user_specified_nameembedding_1/embeddings:3/
-
_user_specified_nameattention_layer/W_a:3/
-
_user_specified_nameattention_layer/U_a:3/
-
_user_specified_nameattention_layer/V_a:.*
(
_user_specified_namedense_1/kernel:,(
&
_user_specified_namedense_1/bias:73
1
_user_specified_namelstm_3/lstm_cell/kernel:A=
;
_user_specified_name#!lstm_3/lstm_cell/recurrent_kernel:5	1
/
_user_specified_namelstm_3/lstm_cell/bias:=
9

_output_shapes
: 

_user_specified_nameConst
ŻI
î
while_body_53126
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_shape_inputs_0;
'while_shape_1_readvariableop_resource_0:
<
(while_matmul_1_readvariableop_resource_0:
:
'while_shape_3_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_shape_inputs9
%while_shape_1_readvariableop_resource:
:
&while_matmul_1_readvariableop_resource:
8
%while_shape_3_readvariableop_resource:	˘while/MatMul_1/ReadVariableOp˘while/transpose/ReadVariableOp˘ while/transpose_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   §
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
element_dtype0]
while/ShapeShapewhile_shape_inputs_0*
T0*
_output_shapes
::íĎ]
while/unstackUnpackwhile/Shape:output:0*
T0*
_output_shapes
: : : *	
num
while/Shape_1/ReadVariableOpReadVariableOp'while_shape_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0^
while/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      _
while/unstack_1Unpackwhile/Shape_1:output:0*
T0*
_output_shapes
: : *	
numd
while/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   
while/ReshapeReshapewhile_shape_inputs_0while/Reshape/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/transpose/ReadVariableOpReadVariableOp'while_shape_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0e
while/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       
while/transpose	Transpose&while/transpose/ReadVariableOp:value:0while/transpose/perm:output:0*
T0* 
_output_shapes
:
f
while/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙z
while/Reshape_1Reshapewhile/transpose:y:0while/Reshape_1/shape:output:0*
T0* 
_output_shapes
:
{
while/MatMulMatMulwhile/Reshape:output:0while/Reshape_1:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
while/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :Z
while/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :§
while/Reshape_2/shapePackwhile/unstack:output:0 while/Reshape_2/shape/1:output:0 while/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:
while/Reshape_2Reshapewhile/MatMul:product:0while/Reshape_2/shape:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/MatMul_1/ReadVariableOpReadVariableOp(while_matmul_1_readvariableop_resource_0* 
_output_shapes
:
*
dtype0¤
while/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙V
while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
while/ExpandDims
ExpandDimswhile/MatMul_1:product:0while/ExpandDims/dim:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
	while/addAddV2while/Reshape_2:output:0while/ExpandDims:output:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Y

while/TanhTanhwhile/add:z:0*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙Y
while/Shape_2Shapewhile/Tanh:y:0*
T0*
_output_shapes
::íĎa
while/unstack_2Unpackwhile/Shape_2:output:0*
T0*
_output_shapes
: : : *	
num
while/Shape_3/ReadVariableOpReadVariableOp'while_shape_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0^
while/Shape_3Const*
_output_shapes
:*
dtype0*
valueB"      _
while/unstack_3Unpackwhile/Shape_3:output:0*
T0*
_output_shapes
: : *	
numf
while/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙   }
while/Reshape_3Reshapewhile/Tanh:y:0while/Reshape_3/shape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 while/transpose_1/ReadVariableOpReadVariableOp'while_shape_3_readvariableop_resource_0*
_output_shapes
:	*
dtype0g
while/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       
while/transpose_1	Transpose(while/transpose_1/ReadVariableOp:value:0while/transpose_1/perm:output:0*
T0*
_output_shapes
:	f
while/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙{
while/Reshape_4Reshapewhile/transpose_1:y:0while/Reshape_4/shape:output:0*
T0*
_output_shapes
:	~
while/MatMul_2MatMulwhile/Reshape_3:output:0while/Reshape_4:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Z
while/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value
B :Y
while/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :Š
while/Reshape_5/shapePackwhile/unstack_2:output:0 while/Reshape_5/shape/1:output:0 while/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:
while/Reshape_5Reshapewhile/MatMul_2:product:0while/Reshape_5/shape:output:0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
while/SqueezeSqueezewhile/Reshape_5:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims

˙˙˙˙˙˙˙˙˙c
while/SoftmaxSoftmaxwhile/Squeeze:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/Softmax:softmax:0*
_output_shapes
: *
element_dtype0:éčŇO
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :`
while/add_1AddV2while_placeholderwhile/add_1/y:output:0*
T0*
_output_shapes
: O
while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_2AddV2while_while_loop_counterwhile/add_2/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_2:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: [
while/Identity_2Identitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: u
while/Identity_4Identitywhile/Softmax:softmax:0^while/NoOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

while/NoOpNoOp^while/MatMul_1/ReadVariableOp^while/transpose/ReadVariableOp!^while/transpose_1/ReadVariableOp*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"R
&while_matmul_1_readvariableop_resource(while_matmul_1_readvariableop_resource_0"P
%while_shape_1_readvariableop_resource'while_shape_1_readvariableop_resource_0"P
%while_shape_3_readvariableop_resource'while_shape_3_readvariableop_resource_0"*
while_shape_inputswhile_shape_inputs_0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?: : : : :˙˙˙˙˙˙˙˙˙: : :˙˙˙˙˙˙˙˙˙: : : 2>
while/MatMul_1/ReadVariableOpwhile/MatMul_1/ReadVariableOp2@
while/transpose/ReadVariableOpwhile/transpose/ReadVariableOp2D
 while/transpose_1/ReadVariableOp while/transpose_1/ReadVariableOp:J F

_output_shapes
: 
,
_user_specified_namewhile/loop_counter:PL

_output_shapes
: 
2
_user_specified_namewhile/maximum_iterations:

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:˙˙˙˙˙˙˙˙˙:EA

_output_shapes
: 
'
_user_specified_namestrided_slice:_[

_output_shapes
: 
A
_user_specified_name)'TensorArrayUnstack/TensorListFromTensor:UQ
-
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource"§L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ü
serving_defaultč
D
input_29
serving_default_input_2:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
<
input_61
serving_default_input_6:0˙˙˙˙˙˙˙˙˙
<
input_71
serving_default_input_7:0˙˙˙˙˙˙˙˙˙
A
input_86
serving_default_input_8:0˙˙˙˙˙˙˙˙˙I
dense_1>
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8;
lstm_31
StatefulPartitionedCall:1˙˙˙˙˙˙˙˙˙=
lstm_3_11
StatefulPartitionedCall:2˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:Í
ą
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
ľ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

embeddings"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ú
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
 cell
!
state_spec"
_tf_keras_rnn_layer
"
_tf_keras_input_layer
Ŕ
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses
(W_a
)U_a
*V_a"
_tf_keras_layer
Ľ
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
ť
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias"
_tf_keras_layer
_
0
91
:2
;3
(4
)5
*6
77
88"
trackable_list_wrapper
_
0
91
:2
;3
(4
)5
*6
77
88"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Á
Atrace_0
Btrace_12
'__inference_model_5_layer_call_fn_53663
'__inference_model_5_layer_call_fn_53693ľ
Ž˛Ş
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults˘
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zAtrace_0zBtrace_1
÷
Ctrace_0
Dtrace_12Ŕ
B__inference_model_5_layer_call_and_return_conditional_losses_53372
B__inference_model_5_layer_call_and_return_conditional_losses_53633ľ
Ž˛Ş
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults˘
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zCtrace_0zDtrace_1
ćBă
 __inference__wrapped_model_52154input_2input_8input_6input_7"
˛
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
,
Eserving_default"
signature_map
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ĺ
Ktrace_02Č
+__inference_embedding_1_layer_call_fn_53801
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zKtrace_0

Ltrace_02ă
F__inference_embedding_1_layer_call_and_return_conditional_losses_53810
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zLtrace_0
*:(
×82embedding_1/embeddings
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
š

Mstates
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ř
Strace_0
Ttrace_1
Utrace_2
Vtrace_32í
&__inference_lstm_3_layer_call_fn_53825
&__inference_lstm_3_layer_call_fn_53840
&__inference_lstm_3_layer_call_fn_53857
&__inference_lstm_3_layer_call_fn_53874Ę
Ă˛ż
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults˘

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zStrace_0zTtrace_1zUtrace_2zVtrace_3
Ä
Wtrace_0
Xtrace_1
Ytrace_2
Ztrace_32Ů
A__inference_lstm_3_layer_call_and_return_conditional_losses_54239
A__inference_lstm_3_layer_call_and_return_conditional_losses_54476
A__inference_lstm_3_layer_call_and_return_conditional_losses_54830
A__inference_lstm_3_layer_call_and_return_conditional_losses_55056Ę
Ă˛ż
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults˘

 
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zWtrace_0zXtrace_1zYtrace_2zZtrace_3
"
_generic_user_object
ř
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_random_generator
b
state_size

9kernel
:recurrent_kernel
;bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
5
(0
)1
*2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
ů
htrace_02Ü
/__inference_attention_layer_layer_call_fn_55070¨
Ą˛
FullArgSpec 
args
jinputs
	jverbose
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zhtrace_0

itrace_02÷
J__inference_attention_layer_layer_call_and_return_conditional_losses_55338¨
Ą˛
FullArgSpec 
args
jinputs
	jverbose
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zitrace_0
':%
2attention_layer/W_a
':%
2attention_layer/U_a
&:$	2attention_layer/V_a
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
ŕ
otrace_02Ă
&__inference_concat_layer_call_fn_55344
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zotrace_0
ű
ptrace_02Ţ
A__inference_concat_layer_call_and_return_conditional_losses_55351
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zptrace_0
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
­
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
á
vtrace_02Ä
'__inference_dense_1_layer_call_fn_55360
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zvtrace_0
ü
wtrace_02ß
B__inference_dense_1_layer_call_and_return_conditional_losses_55391
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zwtrace_0
": 
×82dense_1/kernel
:×82dense_1/bias
+:)
2lstm_3/lstm_cell/kernel
5:3
2!lstm_3/lstm_cell/recurrent_kernel
$:"2lstm_3/lstm_cell/bias
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Bţ
'__inference_model_5_layer_call_fn_53663input_2input_8input_6input_7"Ź
Ľ˛Ą
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Bţ
'__inference_model_5_layer_call_fn_53693input_2input_8input_6input_7"Ź
Ľ˛Ą
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
B__inference_model_5_layer_call_and_return_conditional_losses_53372input_2input_8input_6input_7"Ź
Ľ˛Ą
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
B__inference_model_5_layer_call_and_return_conditional_losses_53633input_2input_8input_6input_7"Ź
Ľ˛Ą
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
#__inference_signature_wrapper_53794input_2input_6input_7input_8"ş
ł˛Ż
FullArgSpec
args 
varargs
 
varkw
 
defaults
 =

kwonlyargs/,
	jinput_2
	jinput_6
	jinput_7
	jinput_8
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŐBŇ
+__inference_embedding_1_layer_call_fn_53801inputs"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
đBí
F__inference_embedding_1_layer_call_and_return_conditional_losses_53810inputs"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
÷Bô
&__inference_lstm_3_layer_call_fn_53825inputs_0"˝
ś˛˛
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
÷Bô
&__inference_lstm_3_layer_call_fn_53840inputs_0"˝
ś˛˛
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
&__inference_lstm_3_layer_call_fn_53857inputsinitial_state_0initial_state_1"˝
ś˛˛
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
&__inference_lstm_3_layer_call_fn_53874inputsinitial_state_0initial_state_1"˝
ś˛˛
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
A__inference_lstm_3_layer_call_and_return_conditional_losses_54239inputs_0"˝
ś˛˛
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
A__inference_lstm_3_layer_call_and_return_conditional_losses_54476inputs_0"˝
ś˛˛
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
˛BŻ
A__inference_lstm_3_layer_call_and_return_conditional_losses_54830inputsinitial_state_0initial_state_1"˝
ś˛˛
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
˛BŻ
A__inference_lstm_3_layer_call_and_return_conditional_losses_55056inputsinitial_state_0initial_state_1"˝
ś˛˛
FullArgSpec:
args2/
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
5
90
:1
;2"
trackable_list_wrapper
5
90
:1
;2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
Ă
}trace_0
~trace_12
)__inference_lstm_cell_layer_call_fn_55408
)__inference_lstm_cell_layer_call_fn_55425ł
Ź˛¨
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z}trace_0z~trace_1
ű
trace_0
trace_12Â
D__inference_lstm_cell_layer_call_and_return_conditional_losses_55567
D__inference_lstm_cell_layer_call_and_return_conditional_losses_55645ł
Ź˛¨
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults˘
p 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ztrace_0ztrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
đBí
/__inference_attention_layer_layer_call_fn_55070inputs_0inputs_1"Ł
˛
FullArgSpec 
args
jinputs
	jverbose
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
J__inference_attention_layer_layer_call_and_return_conditional_losses_55338inputs_0inputs_1"Ł
˛
FullArgSpec 
args
jinputs
	jverbose
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBŮ
&__inference_concat_layer_call_fn_55344inputs_0inputs_1"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
÷Bô
A__inference_concat_layer_call_and_return_conditional_losses_55351inputs_0inputs_1"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŃBÎ
'__inference_dense_1_layer_call_fn_55360inputs"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ěBé
B__inference_dense_1_layer_call_and_return_conditional_losses_55391inputs"
˛
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ýBú
)__inference_lstm_cell_layer_call_fn_55408inputsstates_0states_1"Ž
§˛Ł
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ýBú
)__inference_lstm_cell_layer_call_fn_55425inputsstates_0states_1"Ž
§˛Ł
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
D__inference_lstm_cell_layer_call_and_return_conditional_losses_55567inputsstates_0states_1"Ž
§˛Ł
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
B
D__inference_lstm_cell_layer_call_and_return_conditional_losses_55645inputsstates_0states_1"Ž
§˛Ł
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 __inference__wrapped_model_52154ă	9;:()*78´˘°
¨˘¤
Ą
*'
input_2˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
'$
input_8˙˙˙˙˙˙˙˙˙
"
input_6˙˙˙˙˙˙˙˙˙
"
input_7˙˙˙˙˙˙˙˙˙
Ş "Ş
:
dense_1/,
dense_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8
+
lstm_3!
lstm_3˙˙˙˙˙˙˙˙˙
/
lstm_3_1# 
lstm_3_1˙˙˙˙˙˙˙˙˙ż
J__inference_attention_layer_layer_call_and_return_conditional_losses_55338đ()*r˘o
h˘e
_\
(%
inputs_0˙˙˙˙˙˙˙˙˙
0-
inputs_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "u˘r
k˘h
2/

tensor_0_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2/

tensor_0_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
/__inference_attention_layer_layer_call_fn_55070â()*r˘o
h˘e
_\
(%
inputs_0˙˙˙˙˙˙˙˙˙
0-
inputs_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
p 
Ş "g˘d
0-
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
0-
tensor_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙ú
A__inference_concat_layer_call_and_return_conditional_losses_55351´v˘s
l˘i
gd
0-
inputs_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
0-
inputs_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ":˘7
0-
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Ô
&__inference_concat_layer_call_fn_55344Šv˘s
l˘i
gd
0-
inputs_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
0-
inputs_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "/,
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ĺ
B__inference_dense_1_layer_call_and_return_conditional_losses_5539178=˘:
3˘0
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ":˘7
0-
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8
 
'__inference_dense_1_layer_call_fn_55360t78=˘:
3˘0
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "/,
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8Ă
F__inference_embedding_1_layer_call_and_return_conditional_losses_53810y8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ":˘7
0-
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
+__inference_embedding_1_layer_call_fn_53801n8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "/,
unknown˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˛
A__inference_lstm_3_layer_call_and_return_conditional_losses_54239ě9;:P˘M
F˘C
52
0-
inputs_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p

 
Ş "˘

2/

tensor_0_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
%"

tensor_0_1˙˙˙˙˙˙˙˙˙
%"

tensor_0_2˙˙˙˙˙˙˙˙˙
 ˛
A__inference_lstm_3_layer_call_and_return_conditional_losses_54476ě9;:P˘M
F˘C
52
0-
inputs_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p 

 
Ş "˘

2/

tensor_0_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
%"

tensor_0_1˙˙˙˙˙˙˙˙˙
%"

tensor_0_2˙˙˙˙˙˙˙˙˙
 
A__inference_lstm_3_layer_call_and_return_conditional_losses_54830Â9;:Ľ˘Ą
˘
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p
[X
*'
initial_state_0˙˙˙˙˙˙˙˙˙
*'
initial_state_1˙˙˙˙˙˙˙˙˙
Ş "˘

2/

tensor_0_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
%"

tensor_0_1˙˙˙˙˙˙˙˙˙
%"

tensor_0_2˙˙˙˙˙˙˙˙˙
 
A__inference_lstm_3_layer_call_and_return_conditional_losses_55056Â9;:Ľ˘Ą
˘
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p 
[X
*'
initial_state_0˙˙˙˙˙˙˙˙˙
*'
initial_state_1˙˙˙˙˙˙˙˙˙
Ş "˘

2/

tensor_0_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
%"

tensor_0_1˙˙˙˙˙˙˙˙˙
%"

tensor_0_2˙˙˙˙˙˙˙˙˙
 
&__inference_lstm_3_layer_call_fn_53825Ř9;:P˘M
F˘C
52
0-
inputs_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p

 
Ş "|
0-
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
# 
tensor_1˙˙˙˙˙˙˙˙˙
# 
tensor_2˙˙˙˙˙˙˙˙˙
&__inference_lstm_3_layer_call_fn_53840Ř9;:P˘M
F˘C
52
0-
inputs_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p 

 
Ş "|
0-
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
# 
tensor_1˙˙˙˙˙˙˙˙˙
# 
tensor_2˙˙˙˙˙˙˙˙˙Ů
&__inference_lstm_3_layer_call_fn_53857Ž9;:Ľ˘Ą
˘
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p
[X
*'
initial_state_0˙˙˙˙˙˙˙˙˙
*'
initial_state_1˙˙˙˙˙˙˙˙˙
Ş "|
0-
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
# 
tensor_1˙˙˙˙˙˙˙˙˙
# 
tensor_2˙˙˙˙˙˙˙˙˙Ů
&__inference_lstm_3_layer_call_fn_53874Ž9;:Ľ˘Ą
˘
.+
inputs˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙

 
p 
[X
*'
initial_state_0˙˙˙˙˙˙˙˙˙
*'
initial_state_1˙˙˙˙˙˙˙˙˙
Ş "|
0-
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
# 
tensor_1˙˙˙˙˙˙˙˙˙
# 
tensor_2˙˙˙˙˙˙˙˙˙ĺ
D__inference_lstm_cell_layer_call_and_return_conditional_losses_555679;:˘
y˘v
!
inputs˙˙˙˙˙˙˙˙˙
M˘J
# 
states_0˙˙˙˙˙˙˙˙˙
# 
states_1˙˙˙˙˙˙˙˙˙
p
Ş "˘
˘~
%"

tensor_0_0˙˙˙˙˙˙˙˙˙
UR
'$
tensor_0_1_0˙˙˙˙˙˙˙˙˙
'$
tensor_0_1_1˙˙˙˙˙˙˙˙˙
 ĺ
D__inference_lstm_cell_layer_call_and_return_conditional_losses_556459;:˘
y˘v
!
inputs˙˙˙˙˙˙˙˙˙
M˘J
# 
states_0˙˙˙˙˙˙˙˙˙
# 
states_1˙˙˙˙˙˙˙˙˙
p 
Ş "˘
˘~
%"

tensor_0_0˙˙˙˙˙˙˙˙˙
UR
'$
tensor_0_1_0˙˙˙˙˙˙˙˙˙
'$
tensor_0_1_1˙˙˙˙˙˙˙˙˙
 ˇ
)__inference_lstm_cell_layer_call_fn_554089;:˘
y˘v
!
inputs˙˙˙˙˙˙˙˙˙
M˘J
# 
states_0˙˙˙˙˙˙˙˙˙
# 
states_1˙˙˙˙˙˙˙˙˙
p
Ş "{˘x
# 
tensor_0˙˙˙˙˙˙˙˙˙
QN
%"

tensor_1_0˙˙˙˙˙˙˙˙˙
%"

tensor_1_1˙˙˙˙˙˙˙˙˙ˇ
)__inference_lstm_cell_layer_call_fn_554259;:˘
y˘v
!
inputs˙˙˙˙˙˙˙˙˙
M˘J
# 
states_0˙˙˙˙˙˙˙˙˙
# 
states_1˙˙˙˙˙˙˙˙˙
p 
Ş "{˘x
# 
tensor_0˙˙˙˙˙˙˙˙˙
QN
%"

tensor_1_0˙˙˙˙˙˙˙˙˙
%"

tensor_1_1˙˙˙˙˙˙˙˙˙Ś
B__inference_model_5_layer_call_and_return_conditional_losses_53372ß	9;:()*78ź˘¸
°˘Ź
Ą
*'
input_2˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
'$
input_8˙˙˙˙˙˙˙˙˙
"
input_6˙˙˙˙˙˙˙˙˙
"
input_7˙˙˙˙˙˙˙˙˙
p

 
Ş "˘

2/

tensor_0_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8
%"

tensor_0_1˙˙˙˙˙˙˙˙˙
%"

tensor_0_2˙˙˙˙˙˙˙˙˙
 Ś
B__inference_model_5_layer_call_and_return_conditional_losses_53633ß	9;:()*78ź˘¸
°˘Ź
Ą
*'
input_2˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
'$
input_8˙˙˙˙˙˙˙˙˙
"
input_6˙˙˙˙˙˙˙˙˙
"
input_7˙˙˙˙˙˙˙˙˙
p 

 
Ş "˘

2/

tensor_0_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8
%"

tensor_0_1˙˙˙˙˙˙˙˙˙
%"

tensor_0_2˙˙˙˙˙˙˙˙˙
 ÷
'__inference_model_5_layer_call_fn_53663Ë	9;:()*78ź˘¸
°˘Ź
Ą
*'
input_2˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
'$
input_8˙˙˙˙˙˙˙˙˙
"
input_6˙˙˙˙˙˙˙˙˙
"
input_7˙˙˙˙˙˙˙˙˙
p

 
Ş "|
0-
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8
# 
tensor_1˙˙˙˙˙˙˙˙˙
# 
tensor_2˙˙˙˙˙˙˙˙˙÷
'__inference_model_5_layer_call_fn_53693Ë	9;:()*78ź˘¸
°˘Ź
Ą
*'
input_2˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
'$
input_8˙˙˙˙˙˙˙˙˙
"
input_6˙˙˙˙˙˙˙˙˙
"
input_7˙˙˙˙˙˙˙˙˙
p 

 
Ş "|
0-
tensor_0˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8
# 
tensor_1˙˙˙˙˙˙˙˙˙
# 
tensor_2˙˙˙˙˙˙˙˙˙°
#__inference_signature_wrapper_53794	9;:()*78Ů˘Ő
˘ 
ÍŞÉ
5
input_2*'
input_2˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
-
input_6"
input_6˙˙˙˙˙˙˙˙˙
-
input_7"
input_7˙˙˙˙˙˙˙˙˙
2
input_8'$
input_8˙˙˙˙˙˙˙˙˙"Ş
:
dense_1/,
dense_1˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙×8
+
lstm_3!
lstm_3˙˙˙˙˙˙˙˙˙
/
lstm_3_1# 
lstm_3_1˙˙˙˙˙˙˙˙˙