""" Model zoo for domain detection
"""
from keras.layers import LSTM, Conv1D, GRU, Bidirectional, BatchNormalization, Multiply, merge
from keras.layers import Dense, Activation, Dropout, Input, Flatten, TimeDistributed, Reshape, Permute, RepeatVector, Lambda
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.activations import softmax
# from keras.preprocessing import sequence
from keras.models import Model
import keras.backend as K
from my_keras_layers import *

TIME_STEPS = 1496
# TIME_STEPS = 1496
HIDDEN_SIZE = 256

def gru_1c2r(input_shape):
	# GRU sequence length = 1370
	X_input = Input(shape = input_shape)

	X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	# X = Dropout(0.8)(X)

	X = GRU(units=128, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = GRU(units=128, return_sequences=False)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = Dense(1, activation='sigmoid')(X)

	model = Model(inputs=X_input, outputs=X)
	return model


def gru_2c2r(input_shape):
	# GRU sequence length = 340
	X_input = Input(shape = input_shape)

	X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	# X = Dropout(0.8)(X)

	X = Conv1D(filters=196, kernel_size=15, strides=4)(X)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	# X = Dropout(0.8)(X)

	X = GRU(units=128, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = GRU(units=128, return_sequences=False)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = Dense(1, activation='sigmoid')(X)

	model = Model(inputs=X_input, outputs=X)
	return model


def gru_1c2r1c2fc(input_shape):
	# GRU sequence length = 1370
	X_input = Input(shape = input_shape)

	X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	# X = Dropout(0.8)(X)

	X = GRU(units=128, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = GRU(units=128, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = Conv1D(filters=1, kernel_size=15, strides=4)(X)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	# X = Dropout(0.8)(X)

	# X = TimeDistributed(Flatten())(X)
	# newdim = tuple([340])
	# X = Reshape(newdim)(X)

	X = Reshape((-1,))(X) # drop the dim=1 dimension
	X = Dense(34, activation='relu')(X)
	X = Dense(1, activation='sigmoid')(X)

	model = Model(inputs=X_input, outputs=X)
	return model


def gru_1c2r2fc(input_shape):
	# GRU sequence length = 1370
	X_input = Input(shape = input_shape)

	X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	# X = Dropout(0.8)(X)

	X = GRU(units=128, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = GRU(units=128, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = TimeDistributed(Dense(1, activation = "relu"))(X)
	X = Reshape((-1,))(X)
	X = Dense(128, activation='relu')(X)
	X = Dense(1, activation='sigmoid')(X)

	model = Model(inputs=X_input, outputs=X)
	return model


def gru_1c2rflat2fc(input_shape):
	# GRU sequence length = 1370
	X_input = Input(shape = input_shape)

	X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	# X = Dropout(0.8)(X)

	X = GRU(units=128, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = GRU(units=128, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = Flatten()(X)
	X = Dense(128, activation='relu')(X)
	X = Dense(1, activation='sigmoid')(X)

	model = Model(inputs=X_input, outputs=X)
	return model


def gru_maxpool(input_shape):
	# GRU sequence length = 1370
	X_input = Input(shape = input_shape)

	X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	# X = Dropout(0.8)(X)

	X = GRU(units=128, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = GRU(units=128, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = GlobalMaxPooling1D()(X)
	X = Dense(1, activation='sigmoid')(X)

	model = Model(inputs=X_input, outputs=X)
	return model


def gru_averagepool(input_shape):
	# GRU sequence length = 1370
	X_input = Input(shape = input_shape)

	X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	# X = Dropout(0.8)(X)

	X = GRU(units=128, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = GRU(units=128, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = GlobalAveragePooling1D()(X)
	X = Dense(1, activation='sigmoid')(X)

	model = Model(inputs=X_input, outputs=X)
	return model


def gru_attention_original(input_shape):
	# GRU sequence length = 1370
	X_input = Input(shape = input_shape)

	X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	# X = Dropout(0.8)(X)

	X = GRU(units=HIDDEN_SIZE, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = GRU(units=HIDDEN_SIZE, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	gru = BatchNormalization()(X)

	# mask = TimeDistributed(Dense(128, activation = 'softmax'))(gru)
	# merged = Multiply()([gru, mask])
	# # merged = merge([gru, mask], mode='mul')
	# merged = Flatten()(merged)

	attention_mul = _attention_original(gru)
	attention_mul = Flatten()(attention_mul)

	X = Dense(128, activation='relu')(attention_mul)
	X = Dense(1, activation='sigmoid')(X)

	model = Model(inputs=X_input, outputs=X)
	return model


def gru_attention(input_shape):
	# GRU sequence length = 1370
	X_input = Input(shape = input_shape)

	X = Conv1D(filters=196, kernel_size=15, strides=4, activation='relu')(X_input)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	# X = Dropout(0.8)(X)

	X = GRU(units=HIDDEN_SIZE, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = GRU(units=HIDDEN_SIZE, return_sequences=True)(X)
	# X = Dropout(0.8)(X)
	gru = BatchNormalization()(X)

	attention_mul = _attention_yoshua_5head(gru)
	# attention_mul = _attention_mul(gru)
	# attention_mul = AttentionWithContext()(gru)
	X = Dense(1, activation='sigmoid')(attention_mul)

	model = Model(inputs=X_input, outputs=X)
	return model


def bigru_attention(input_shape):
	# GRU sequence length = 1370
	X_input = Input(shape = input_shape)

	X = Conv1D(filters=196, kernel_size=15, strides=4, activation='relu')(X_input)
	X = BatchNormalization()(X)
	X = Activation('relu')(X)
	# X = Dropout(0.8)(X)

	X = Bidirectional(GRU(units=HIDDEN_SIZE, return_sequences=True))(X)
	# X = Dropout(0.8)(X)
	X = BatchNormalization()(X)

	X = Bidirectional(GRU(units=HIDDEN_SIZE, return_sequences=True))(X)
	# X = Dropout(0.8)(X)
	gru = BatchNormalization()(X)

	attention_mul = _attention_yoshua_1head(gru, 2)
	# attention_mul = _attention_mul(gru)
	# attention_mul = AttentionWithContext()(gru)
	X = Dense(1, activation='sigmoid')(attention_mul)

	model = Model(inputs=X_input, outputs=X)
	return model


def _attention_original(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    # each hidden vector doesn't multiply by a scalar, but a vector with different values at different positions.
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul


def _attention_mul(inputs):
	# inputs.shape = (batch_size, time_steps, input_dim)
	# one layer neural network to calculate attention
	a_probs = Dense(1, activation='tanh')(inputs)
	a_probs = Flatten()(a_probs)
	# a_probs = Multiply(K.sqrt(HIDDEN_SIZE))(a_probs)
	# a_probs = Lambda(lambda x: x / 16)(a_probs)
	a_probs = Activation('softmax')(a_probs)

	mask = RepeatVector(HIDDEN_SIZE)(a_probs)
	mask = Permute((2,1))(mask)

	output_attention_mul = merge([inputs, mask], name='attention_mul', mode='mul')
	output = Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)
	return output


def _attention_yoshua_1head(inputs, bi=1):
	# inputs.shape = (batch_size, time_steps, input_dim)
	# two layer neural network to calculate attention
	a_probs = Dense(16, activation='tanh')(inputs)  # W1
	a_probs = Dense(1)(a_probs)                    # W2
	a_probs = Flatten()(a_probs)
	# a_probs.shape = (batch_size, time_steps)
	a_probs = Activation('softmax')(a_probs)

	mask = RepeatVector(HIDDEN_SIZE*bi)(a_probs)
	mask = Permute((2,1))(mask)

	output_attention_mul = merge([inputs, mask], name='attention_mul', mode='mul')
	output = Lambda(lambda x: K.sum(x, axis=1))(output_attention_mul)
	return output

# hasn't work out the matrix mul
def _attention_yoshua_5head(inputs, num_head=10):
	# inputs.shape = (batch_size, time_steps, input_dim)
	a_probs = Dense(16, activation='tanh')(inputs)  # W1
	a_probs = Dense(num_head)(a_probs)              # W2
	a_probs = Lambda(lambda x: softmax(x, axis=-2))(a_probs)

	output = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1,1]))([a_probs, inputs])
	output = Flatten()(output)
	return output