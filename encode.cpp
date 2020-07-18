#include "encode.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <mkl.h>
#include <string.h>
using namespace std;


NMT::Encoder::Encoder(size_t& head_num,
		      size_t& hidden_num,
		      size_t& layer_num,
		      size_t& vocabe_size,
		      size_t& filter_size, 
		      vector<vector<vector<float>>>& weight,
		      vector<float>& weight_embedding,
		      vector<float>& weight_language,
		      vector<float>& weight_scale,
		      vector<float>& weight_bias)
{
	              this->head_num = head_num;
	              this->hidden_num = hidden_num;
	              this->layer_num = layer_num;
	              this->vocabe_size = vocabe_size;
	              this->filter_size = filter_size;
	              this->weight = weight;
	              this->weight_embedding = weight_embedding;
	              this->weight_language = weight_language;
	              this->weight_scale = weight_scale;
	              this->weight_bias = weight_bias;
}

void NMT::Encoder::EmbeddingLookup(const int* input, const size_t& batch_size, const size_t& length, vector<float>& embedding_word, vector<int>& mask, vector<int>& target_language_id)
/*
* length: the lengh of a sentence
*/
{
	vector<float> zero(hidden_num, 0.0);
	for (int i = 0; i < batch_size * length; i++)
	{
		if (mask[i]==1)
		{
		vector<float>::iterator begin = weight_embedding.begin() + hidden_num * input[i];
		embedding_word.insert(embedding_word.end(), begin, begin + hidden_num);
		}
		else
		{
		embedding_word.insert(embedding_word.end(), zero.begin(), zero.end());
		};
                //cout<<*begin<<endl;
	}
	//don't know why, we can merge this to weight when online
	for(auto& info:embedding_word)
	{
		info *= 32.0;// a magic number 
	}
	//change embedding by language id
	//vector<int> target_language_id(batch_size, 1);
	ChangeEmbedding(embedding_word, batch_size, length, target_language_id);

}

void NMT::Encoder::ChangeEmbedding(vector<float>& embedding_word, const size_t& batch_size, const size_t& length,  vector<int>& target_language_id)
/*
 * chansge embedding by langue id 
*/
{
	int size = length*hidden_num;
	for(int i=0; i<batch_size; i++)
	{
        	
		vector<float>::iterator begin = weight_language.begin() + hidden_num * target_language_id[i];
		for(int j=0; j<size; j++)
		{
			embedding_word[i*size + j] += *(begin + (j%hidden_num));
		}
	}
}

void NMT::Encoder::LayerPreprocess(vector<float>& layer_input, const size_t& batch_size, const size_t& length, const float* scale, const float* bias)
/*
* layer_input:|1,2,.., 1024| ,|2,3..., 1025|.....|n, n+1, ..., n+1023|, n = batch_size * length
* length:the length of a sentence
*/
{
	//epsilon
	float epsilon = 1E-6;
	vector<float>::iterator begin = layer_input.begin();

	for(int i=0; i < batch_size * length; i++ )
	{
		//mean
		float sum = accumulate(begin, begin + hidden_num, 0.0);
		float mean = sum / hidden_num;
		//variance
		float accum = 0.0;
		for_each(begin, begin + hidden_num, [&](const float& d) {accum += (d - mean) * (d - mean); });
		float variance = accum / hidden_num;
		//norm
		for_each(begin, begin + hidden_num, [&](float& d) { d = (d - mean) / sqrt(variance + epsilon); });
		//mut and add
		for (int j = 0; j < hidden_num; j++, begin++)
		{
			*begin = (*begin) * scale[j] + bias[j];
		}
	}

}
void NMT::Encoder::LayerPostprocess(vector<float>& layer_input, const vector<float>& temp)
{
	for (int i = 0; i < layer_input.size(); i++)
	{
		layer_input[i] = layer_input[i] + temp[i];
	}
}

void NMT::Encoder::GetPositionX(const float* position_embedding, const size_t max_length, const size_t& length, vector<float>& position_x)
{
	int max = 2 * max_length;
	vector<int> mat(length * length);
	//get position and encode
	for (int i = 0; i < length * length; i++ )
	{
		//get position
		int tmp = i % length - (i / length) + max_length;
		mat[i] = tmp > max? max:tmp;
		if (tmp < 0) mat[i] = 0;
		//cout<<mat[i]<<" ";

		//get encode
		//vector<float>::const_iterator begin = position_embedding + hidden_num / head_num * mat[i];
		const float* begin = position_embedding + hidden_num / head_num * mat[i];
		position_x.insert(position_x.end(), begin, begin + hidden_num / head_num);
	}
}

void NMT::Encoder::MulPositionKey(const size_t& batch_size, const size_t& length, float* input, float* position_key, float* out)
{

#define GRP_COUNT 1

	MKL_INT    b_m[GRP_COUNT] = { head_num };
	MKL_INT    b_k[GRP_COUNT] = { hidden_num / head_num };
	MKL_INT    b_n[GRP_COUNT] = { length };

	MKL_INT    lda[GRP_COUNT] = { hidden_num / head_num };
	MKL_INT    ldb[GRP_COUNT] = { hidden_num / head_num };
	MKL_INT    ldc[GRP_COUNT] = { length };


	CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
	CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasTrans };
	float    b_alpha[GRP_COUNT] = { 1.0 };
	float    b_beta[GRP_COUNT] = { 0.0 };

	const MKL_INT    size_per_grp[GRP_COUNT] = { length * batch_size };

	vector<float*> a_array(length * batch_size);
	vector<float*> b_array(length * batch_size);
	vector<float*> c_array(length * batch_size);
	for (int i = 0; i < batch_size; ++i)
	{
		for (int j = 0; j < length; j++)
		{
			a_array[i * length + j] = input + i * length * hidden_num + j * hidden_num;
			b_array[i * length + j] = position_key+ j * (hidden_num / head_num) * length;
			c_array[i * length + j] = out + i * length * (length * head_num) + j * length * head_num;
		}
	}
	cblas_sgemm_batch(CblasRowMajor, transA, transB,
		b_m, b_n, b_k, b_alpha,
		const_cast<const float**>(a_array.data()), lda,
		const_cast<const float**>(b_array.data()), ldb, b_beta,
		c_array.data(), ldc,
		GRP_COUNT, size_per_grp);
}
void NMT::Encoder::MulPositionValue(const size_t& batch_size, const size_t& length, float*input, float* position_val, float* out)
{
	//#define GRP_COUNT 1

	//MKL_INT    b_m[GRP_COUNT] = { head_num };
	//MKL_INT    b_k[GRP_COUNT] = { length };
	//MKL_INT    b_n[GRP_COUNT] = { hidden_num / head_num };

	//MKL_INT    lda[GRP_COUNT] = { head_num };
	//MKL_INT    ldb[GRP_COUNT] = { hidden_num / head_num};
	//MKL_INT    ldc[GRP_COUNT] = { hidden_num / head_num};


	//CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
	//CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasNoTrans };
	//float    b_alpha[GRP_COUNT] = { 1.0 };
	//float    b_beta[GRP_COUNT] = { 0.0 };

	//const MKL_INT    size_per_grp[GRP_COUNT] = { length * batch_size };

	//vector<float*> a_array(length * batch_size);
	//vector<float*> b_array(length * batch_size);
	//vector<float*> c_array(length * batch_size);
	//for (int i = 0; i < batch_size * length; ++i)
	//{
	//	a_array[i] = input + i * head_num * length;
	//	b_array[i] = position_val + (i % length) * (hidden_num / head_num) * length;
	//	c_array[i] = out + i * hidden_num;
	//}
	//cblas_sgemm_batch(CblasRowMajor, transA, transB,
	//	b_m, b_n, b_k, b_alpha,
	//	const_cast<const float**>(a_array.data()), lda,
	//	const_cast<const float**>(b_array.data()), ldb, b_beta,
	//	c_array.data(), ldc,
	//	GRP_COUNT, size_per_grp);
	#define GRP_COUNT 1

	MKL_INT    b_m[GRP_COUNT] = { 1 };
	MKL_INT    b_k[GRP_COUNT] = { length };
	MKL_INT    b_n[GRP_COUNT] = { hidden_num / head_num };

	MKL_INT    lda[GRP_COUNT] = { length * head_num };
	MKL_INT    ldb[GRP_COUNT] = { hidden_num / head_num};
	MKL_INT    ldc[GRP_COUNT] = { hidden_num / head_num};


	CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
	CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasNoTrans };
	float    b_alpha[GRP_COUNT] = { 1.0 };
	float    b_beta[GRP_COUNT] = { 0.0 };

	const MKL_INT    size_per_grp[GRP_COUNT] = { batch_size * length * head_num };

	vector<float*> a_array(batch_size * length * head_num);
	vector<float*> b_array(batch_size * length * head_num);
	vector<float*> c_array(batch_size * length * head_num);
	for (int i = 0; i < batch_size * head_num * length; ++i)
	{
		a_array[i] = input + i * length;
		b_array[i] = position_val + length * (hidden_num/head_num) * (i/head_num%length);
		c_array[i] = out + i * (hidden_num/head_num);
	}
	cblas_sgemm_batch(CblasRowMajor, transA, transB,
		b_m, b_n, b_k, b_alpha,
		const_cast<const float**>(a_array.data()), lda,
		const_cast<const float**>(b_array.data()), ldb, b_beta,
		c_array.data(), ldc,
		GRP_COUNT, size_per_grp);
	
}
void NMT::Encoder::SetZero(const size_t& batch_size, const size_t& length, float* input, int* mask)
{
	int size = batch_size * length;
	for(int i = 0; i < size; i++)
	{
		if(mask[i]==1) continue;
		float* begin = input + i * hidden_num;
		float* end = begin + hidden_num;
		for_each(begin, end, [&](float& d) {d = 0.0;});
	}
}
void NMT::Encoder::BuildBias(const size_t& batch_size, const size_t& length,  int* mask, float* bias)
{
	for (int i = 0; i < batch_size*length; i++)
	{
		bias[i] *= (1-mask[i]);
	}
}
void NMT::Encoder::AddBias(float* input, const float* bias, const size_t& batch_size, const size_t& length)
{
	int head_num = 16;
	int one_batch_length = length * length * head_num;
	for (int i = 0; i < batch_size; i++)
	{
		const float* begin_bias = bias + i * length;
		float* begin_input = input + i * one_batch_length;
		for (int j = 0; j < one_batch_length; j++)
		{
			begin_input[j] += begin_bias[j % length];
		}
	}
}
void NMT::Encoder::Attention(float* layer_input, const size_t& batch_size, const size_t& length, const float* q_weight, const float* k_weight, const float* v_weight, const float* key_weight, const float* value_weight, const float* weight, const float* bias, float* output)
{
	MKL_INT m = batch_size * length;
	MKL_INT k = hidden_num;
	MKL_INT n = hidden_num;
	float alpha = 1.0;
	float beta = 0.0;

	//compute q,k,v
	vector<float> tem_q(m * n, 0.0);
	vector<float> tem_k(m * n, 0.0);
	vector<float> tem_v(m * n, 0.0);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, alpha,
		layer_input, k,
		q_weight, n, beta,
		tem_q.data(), n);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, alpha,
		layer_input, k,
		k_weight, n, beta,
		tem_k.data(), n);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, alpha,
		layer_input, k,
		v_weight, n, beta,
		tem_v.data(), n);
	// compute q/sqrt(64)
	float q_mul_num = (1.0 / sqrt(hidden_num / head_num)); 
	for_each(tem_q.begin(), tem_q.end(), [&](float& d) {d *= q_mul_num;});
	//split_head

	//dot_product_attention_relative
	#define GRP_COUNT 1

	// the result of q * k
	vector<float> tem_q_k(batch_size * head_num * length * length, 0.0);

	MKL_INT    b_m[GRP_COUNT] = { length };
	MKL_INT    b_k[GRP_COUNT] = { hidden_num / head_num };
	MKL_INT    b_n[GRP_COUNT] = { length };

	MKL_INT    lda[GRP_COUNT] = { hidden_num };
	MKL_INT    ldb[GRP_COUNT] = { hidden_num };
	MKL_INT    ldc[GRP_COUNT] = { length * head_num };


	CBLAS_TRANSPOSE    transA[GRP_COUNT] = { CblasNoTrans };
	CBLAS_TRANSPOSE    transB[GRP_COUNT] = { CblasTrans };
	float    b_alpha[GRP_COUNT] = { 1.0 };
	float    b_beta[GRP_COUNT] = { 0.0 };

	const MKL_INT    size_per_grp[GRP_COUNT] = { head_num * batch_size };

	vector<float*> a_array(head_num * batch_size);
	vector<float*> b_array(head_num * batch_size);
	vector<float*> c_array(head_num * batch_size);

	for (int i = 0; i < batch_size; ++i) 
	{
             for (int j = 0; j < head_num; j++)
	     {
		a_array[i*head_num+j] = tem_q.data() + i * length * hidden_num + j * (hidden_num/head_num);
		b_array[i*head_num+j] = tem_k.data() + i * length * hidden_num + j * (hidden_num/head_num);
		c_array[i*head_num+j] = tem_q_k.data() + i * length * (head_num*length) + j * length;
             }
	}

	cblas_sgemm_batch(CblasRowMajor, transA, transB, 
		b_m, b_n, b_k, b_alpha,
		const_cast<const float**>(a_array.data()), lda,
		const_cast<const float**>(b_array.data()), ldb, b_beta,
		c_array.data(), ldc,
		GRP_COUNT, size_per_grp);
	//get relative_position_key 
	vector<float> position_key;
	vector<float> position_q(length * length * head_num * batch_size);
        GetPositionX(key_weight, 20, length, position_key);
	// the result of q * position
	MulPositionKey(batch_size, length, tem_q.data(), position_key.data(), position_q.data());
	// the result of tem_q_k + position_q 
	for(int i=0; i<tem_q_k.size(); i++)
	{
		tem_q_k[i] += position_q[i];
	}
	// add attention_bias_ignore_padding
	AddBias(tem_q_k.data(), bias, batch_size, length);
	// softmax
	BatchSoftmax(tem_q_k.data(), b_n[0], head_num, batch_size, length);
	//softamx * v

	b_m[0] = length;
	b_k[0] = length;
	b_n[0] = hidden_num / head_num;

	lda[0] = length * head_num;
	ldb[0] = hidden_num;
	ldc[0] = hidden_num;

	transA[0] = CblasNoTrans;
	transB[0] = CblasNoTrans;
	b_alpha[0] = { 1.0 };
	b_beta[0] = { 0.0 };


	for (int i = 0; i <  batch_size; ++i)
        {
            for (int j = 0; j < head_num; j++)
            {
		a_array[i*head_num+j] = tem_q_k.data() + i * length * (head_num*length) + j * length;
		b_array[i*head_num+j] = tem_v.data() + i * length * hidden_num + j * (hidden_num/head_num);
		c_array[i*head_num+j] = tem_q.data() + i * length * hidden_num + j * (hidden_num/head_num);
            }
	}
	cblas_sgemm_batch(CblasRowMajor, transA, transB,
		b_m, b_n, b_k, b_alpha,
		const_cast<const float**>(a_array.data()), lda,
		const_cast<const float**>(b_array.data()), ldb, b_beta,
		c_array.data(), ldc,
		GRP_COUNT, size_per_grp);
	
	// get position_value
	vector<float> position_value;
	vector<float> position_v(batch_size * length * hidden_num);
        GetPositionX(value_weight, 20, length, position_value);
	// the result of softmax * position_value
	MulPositionValue(batch_size, length, tem_q_k.data(), position_value.data(), position_v.data());
	// the result of tem_q + position_v
	for(int i=0; i<tem_q.size(); i++)
	{
		tem_q[i] += position_v[i];
	}
	//last dense
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, alpha,
		tem_q.data(), k,
		weight, n, beta,
		output, n);
};
void NMT::Encoder::BatchSoftmax(float* input_qk, int k, int head_num, const size_t& batch_size, const size_t& length)
/*
 * |1,2,3|4,5,6|7,8,9|....|2,3,4| is the result of q*k, head_num is 16
 * k is 3 , the number of word
*/
{
	for (int i = 0; i < head_num * batch_size * length; i++)
	{
		float* data = input_qk + i * k;
		float sum = 0.0;
		# pragma omp simd reduction(+:sum)
		for (int j = 0; j < k; j++)
		{
			data[j] = exp(data[j]);
			sum += data[j];
		}
		# pragma omp simd
		for (int j = 0; j < k; j++)
		{
			data[j] /= sum;
		}
	}
}
void NMT::Encoder::FeedForward(const vector<float>& input, vector<float>& output, const size_t& batch_size, const size_t& length,int filter, const float* weight, float* bias, string activation)
{
	MKL_INT m = batch_size * length;
	MKL_INT k = input.size()/m;
	MKL_INT n = filter;
	float alpha = 1.0;
	float beta = 1.0;
	//vector<float> tem_q(m * n, 0.0);
	for (int i = 0; i < batch_size * length; i++)
	{
		memcpy(output.data() + i * n , bias, n * 4);//the byte of float is four times to char
	}

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		m, n, k, alpha,
		input.data(), k,
		weight, n, beta,
		output.data(), n);
	if (activation == "relu")
	{
		float min = 0.0;
		int size = m * n;
		for (int i = 0; i < size; i++)
		{
			output[i] = max(min, output[i]);
		}
	}

};
vector<float> NMT::Encoder::Encode(vector<int>& input, const size_t& batch_size, const size_t& length, vector<int>& mask, vector<int>& language_id)
{
	vector<float> embedding_word;
	EmbeddingLookup(input.data(), batch_size, length, embedding_word, mask, language_id);
	vector<float> bias(batch_size*length, -1e9);	
	BuildBias(batch_size, length, mask.data(), bias.data());
	//cout<<"******after embedding************:"<<embedding_word[0]<<" "<<embedding_word[1024]<<" "<<embedding_word[16*1024-1]<<endl;
	for (int i = 0; i < layer_num; i++)
	{
		//self-attention
		vector<float> attention_out = embedding_word;
		LayerPreprocess(attention_out, batch_size, length, weight[i][0].data(), weight[i][1].data());
		Attention(attention_out.data(), batch_size, length, weight[i][2].data(), weight[i][3].data(), weight[i][4].data(), weight[i][12].data(), weight[i][13].data(), weight[i][5].data(), bias.data(), attention_out.data());
		LayerPostprocess(embedding_word, attention_out);

		//ffn
		attention_out = embedding_word;
		vector<float> ffn_out_1(batch_size*length*filter_size, 0.0);
		LayerPreprocess(attention_out, batch_size, length, weight[i][6].data(), weight[i][7].data());
		FeedForward(attention_out, ffn_out_1, batch_size, length, filter_size, weight[i][8].data(), weight[i][9].data(), "relu");
		FeedForward(ffn_out_1, attention_out, batch_size, length, hidden_num, weight[i][10].data(), weight[i][11].data(), "none");
		SetZero(batch_size, length, attention_out.data(), mask.data());
		LayerPostprocess(embedding_word, attention_out);
                
	};
	LayerPreprocess(embedding_word, batch_size, length, weight_scale.data(), weight_bias.data());
        cout<<embedding_word[0]<<" "<<embedding_word[1024]<<endl;
        return embedding_word;//encode_out
}
