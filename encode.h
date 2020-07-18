#ifndef _ENCODE_
#define _ENCODE_

#include <vector>
#include <string>
using namespace std;


namespace NMT
{
	class Encoder
	{
	private:
		size_t head_num;
		size_t hidden_num;
		size_t layer_num;
		size_t vocabe_size;
		size_t filter_size;
		vector<vector<vector<float>>> weight;
		vector<float> weight_embedding;
		vector<float> weight_language;
		vector<float> weight_scale;
		vector<float> weight_bias;

	public:
		Encoder( size_t& head_num,
			 size_t& hidden_num, 
			 size_t& layer_num,
			 size_t& vocabe_size, 
			 size_t& filter_size,
			 vector<vector<vector<float>>>& weight,
			 vector<float>& weight_embedding, 
			 vector<float>& weight_language,
			 vector<float>& weight_scale,
			 vector<float>& weight_bias);

		void SetZero(const size_t& batch_size, const size_t& length, float* input, int* mask);
                void BuildBias(const size_t& batch_size, const size_t& length,  int* mask, float* bias);
		void AddBias(float* input, const float* bias, const size_t& batch_size, const size_t& length);
	        void GetPositionX(const float* position_embedding, const size_t max_length, const size_t& length, vector<float>& position_x);
		void MulPositionKey(const size_t& batch_size, const size_t& length, float* input, float* position_key, float* out);
		void MulPositionValue(const size_t& batch_size, const size_t& length, float*input, float* position_val, float* out);
		void BatchSoftmax(float* input_qk, int k, int head_num, const size_t& batch_size, const size_t& length);
		void EmbeddingLookup(const int*  input, const size_t& batch_size, const size_t& length, vector<float>& embedding_word, vector<int>& mask, vector<int>& target_language_id);
                void ChangeEmbedding(vector<float>& embedding_word, const size_t& batch_size, const size_t& length,  vector<int>& target_language_id);
		void LayerPreprocess(vector<float>& layer_input, const size_t& batch_size, const size_t& length, const float* scale, const float* bias);
		void LayerPostprocess(vector<float>& layer_input, const vector<float>& temp);
		void Attention(float* layer_input, const size_t& batch_size, const size_t& length, const float* q_weight, const float* k_weight, const float* v_weight,
			       const float* key_weight, const float* value_weight, const float* weight, const float* bias, float* output);
		void FeedForward(const vector<float>& input, vector<float>& output, const size_t& batch_size, const size_t& length, int filter, const float* weight, float* bias, string activation);
		vector<float> Encode(vector<int>& input, const size_t& batch_size, const size_t& length, vector<int>& mask, vector<int>& language_id);
	};
}
#endif
