#include <iostream>
#include <iomanip>
#include <mkl.h>
#include <memory>
#include <vector>
#include <assert.h>
#include <numeric>
#include <string>
#include <algorithm>
#include <time.h>
#include <fstream>
#include <chrono>
#include <fstream>
#include "encode.h"

using namespace std;
using namespace NMT;

void load(vector<float>& weight, string dir)
{
	ifstream input(dir);

	if (input.fail())
	{
		cout << "File does not exist" << endl;
		cout << "Exit program" << endl;
		return;
	}
	float num = 0.0;
	while (input >> num)  // 当没有读到文件结尾
	{
		weight.push_back(num);
        	//cout << num << endl;
	}
	input.close();

}

void load_layer_weight(vector<vector<float>>& layer_weight, int num)
{
	cout << "start read layer " << num << " weight" << endl;
	vector<float> layer_self_scale;//0
	vector<float> layer_self_bias;//1
	vector<float> layer_self_q;//2
	vector<float> layer_self_k;//3
	vector<float> layer_self_v;//4
	vector<float> layer_self_last;//5


	vector<float> layer_ffn_scale;//6
	vector<float> layer_ffn_bias;//7
	vector<float> layer_ffn_first_weight;//8
	vector<float> layer_ffn_first_bias;//9
	vector<float> layer_ffn_second_weight;//10
	vector<float> layer_ffn_second_bias;//11
	
	vector<float> layer_self_position_key;//12
	vector<float> layer_self_position_value;//13


	cout << "...:load self attention weight" << endl;
	string name = "./weight/layer_" + to_string(num);
	load(layer_self_scale, name + "_self_scale.txt");
	load(layer_self_bias, name + "_self_bias.txt");
	load(layer_self_q, name + "_self_q.txt");
	load(layer_self_k, name + "_self_k.txt");
	load(layer_self_v, name + "_self_v.txt");
	load(layer_self_last, name + "_self_last.txt");
	load(layer_self_position_key, name + "_self_position_key.txt");
	load(layer_self_position_value, name + "_self_position_value.txt");

	cout << "...:load read fnn weight" << endl;
	load(layer_ffn_scale, name + "_ffn_scale.txt");
	load(layer_ffn_bias, name + "_ffn_bias.txt");
	load(layer_ffn_first_weight, name + "_ffn_first_weight.txt");
	load(layer_ffn_first_bias, name + "_ffn_first_bias.txt");
	load(layer_ffn_second_weight, name + "_ffn_second_weight.txt");
	load(layer_ffn_second_bias, name + "_ffn_second_bias.txt");

	layer_weight.push_back(layer_self_scale);
	layer_weight.push_back(layer_self_bias);
	layer_weight.push_back(layer_self_q);
	layer_weight.push_back(layer_self_k);
	layer_weight.push_back(layer_self_v);
	layer_weight.push_back(layer_self_last);

	layer_weight.push_back(layer_ffn_scale);
	layer_weight.push_back(layer_ffn_bias);
	layer_weight.push_back(layer_ffn_first_weight);
	layer_weight.push_back(layer_ffn_first_bias);
	layer_weight.push_back(layer_ffn_second_weight);
	layer_weight.push_back(layer_ffn_second_bias);

	layer_weight.push_back(layer_self_position_key);
	layer_weight.push_back(layer_self_position_value);
	
	cout << "...:end layer " << num << " weight" << endl;
}



void GetPositionEncode(const vector<float> weight_position_x, const size_t max_length, const size_t& length, string name="position.txt")
{
	int head_num = 16;
	int hidden_num = 1024;
	vector<float> position_encode;

	int max = 2 * max_length;
	vector<int> mat(length * length);
	//get position and encode
	for (int i = 0; i < length * length; i++ )
	{
		//get position
		int tmp = i % length - (i / length) + max_length;
		mat[i] = tmp > max? max:tmp;
		if (tmp < 0) mat[i] = 0;

		//get encode
		vector<float>::const_iterator begin = weight_position_x.begin() + hidden_num / head_num * mat[i];
		position_encode.insert(position_encode.end(), begin, begin + hidden_num);
	}
	//save weight position
	//ofstream f;
	//f.open(name);
	//for (auto info : position_encode)
	//{
	//	f << info << endl;
	//};
	//f.close();
}

int main()
{
	//参数
	size_t head = 16;
	size_t hidden = 1024;
	size_t layer = 6;
	size_t vocab_num = 32768;
	size_t ffn = 4096;
	//导入参数
	cout << ">>start load embedding" << endl;
	vector<float> weight_embedding;
	load(weight_embedding, "./weight/embedding.txt");
	vector<float> weight_language;
	load(weight_language, "./weight/language_embedding.txt");
	cout << "<<end load embedding" << endl;
	cout << ">>start load layer weigth" << endl;
	vector<vector<vector<float>>> weight(layer);
	for(int i = 0; i<layer; i++)
	{
		load_layer_weight(weight[i],i);
	}
	cout << "<<end load embedding" << endl;
	cout << ">>start load last scale/bias" << endl;
	vector<float> weight_scale;
	load(weight_scale, "./weight/scale.txt");
	vector<float> weight_bias;
	load(weight_bias, "./weight/bias.txt");
	cout << ">>end load last scale/bias" << endl;
	//设定输入
	vector<int> input = {115, 29, 112, 18, 17036, 0, 0, 0,   177, 6716, 7667,  9643, 8, 124, 0, 0};
	vector<int> mask = {1,1,1,1,1,0,0,0,   1,1,1,1,1,1,0,0};
	vector<int> language = {1, 1};
	Encoder encoder = Encoder(head,
				hidden,
				layer,
				vocab_num,
				ffn,
				weight,
				weight_embedding,
				weight_language,
				weight_scale,
				weight_bias);
        
	auto time1 = chrono::steady_clock::now();
	vector<float> result = encoder.Encode(input, 2, 8, mask, language);
	auto time2 = chrono::steady_clock::now();
        ofstream f;
	f.open("encode_out.txt");	
	for(auto info:result)
	{
		f<<info<<endl;
	};
	f.close();
	cout << " *************encode time:" << (chrono::duration_cast<chrono::duration<double>>(time2 - time1)).count() << endl;
	return 0;
}
