#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <unordered_map>
	

struct Word {
	int id_;
	int count_;
	Word(int id, int count):id_(id), count_(count){}
};

struct Document {
	int length_;
	int max_id_;
	std::vector<Word* > words_;

	int load(const std::string& line) { // load doc from a line
		std::istringstream istream(line);
		istream >> length_;
		length_ = 0;
		int max_id = -1;
		std::string temp;
		while (istream >> temp) {
			int id, count;
			sscanf(temp.c_str(), "%d:%d", &id, &count);
			Word* newword = new Word(id, count);
			words_.push_back(newword);
			length_ += count;
			max_id = max_id > id ? max_id : id;
		}
		max_id_ = max_id;
		return max_id_+1;
	} 
};

class LDA {
	typedef std::vector<int> VecInt;
	typedef std::vector<double> VecFloat;
public:
	LDA(int num_topic = 100, int num_burn_in = 1000, int num_sample = 50, std::string model_name = "lda", double alpha = 0.5, double beta = 0.1):
		num_topic_(num_topic), num_burn_in_(num_burn_in), num_sample_(num_sample), model_name_(model_name), alpha_(alpha), beta_(beta) 
		{}
	~LDA() {}
	void init() {
		std::cout << "Begin initialization" << std::endl;
		size_t n_docs = corpus_.size();

		if (n_docs == 0) return;

		z_.resize(n_docs);
		for (int i = 0; i < z_.size(); ++i) z_[i].resize(corpus_[i]->length_);

		theta_.resize(n_docs);
		for (auto& vec : theta_) vec.resize(num_topic_);

		phi_.resize(num_topic_);
		for (auto& vec : phi_) vec.resize(num_word_);

		count_doc_topic_.resize(n_docs);
		for (auto& vec : count_doc_topic_) vec.resize(num_topic_);

		count_topic_word_.resize(num_topic_);
		for (auto& vec : count_topic_word_) vec.resize(num_word_);

		sum_doc_word_.resize(n_docs);
		sum_topic_word_.resize(num_topic_);

		std::default_random_engine eng(::time(NULL));
		std::uniform_real_distribution<double> rng(0.0, 0.99999);

		//init 
		for (int m = 0; m < n_docs; ++m) { //for document m
			int n = 0;
			for (int v = 0; v < corpus_[m]->words_.size(); ++v) { 
				int word_id = corpus_[m]->words_[v]->id_;
				for (int k = 0; k < corpus_[m]->words_[v]->count_; ++k) { //for word n
					// sample z_m,n randomly
					double rand = rng(eng);
					z_[m][n] = (int)(rand * num_topic_);
					int topic_id = z_[m][n];
					++count_doc_topic_[m][topic_id];
					++count_topic_word_[topic_id][word_id];
					++sum_topic_word_[topic_id];
					++n;
				}
			}
			sum_doc_word_[m] += n;
		}

		sum_alpha_ = num_topic_ * alpha_;
		sum_beta_ = num_word_ * beta_;

		p_.resize(num_topic_);

		std::cout << "Finish the initialization" << std::endl;

	}
	int train() {
		int iter_time = 0;
		int sample_time = 0;
		while (true) {
			for (int m = 0; m < corpus_.size(); ++m) { // for all document m
				int n = 0;
				for (int v = 0; v < corpus_[m]->words_.size(); ++v) {
					int word_id = corpus_[m]->words_[v]->id_;
					for (int k = 0; k < corpus_[m]->words_[v]->count_; ++k) { // for all word n, w_mn = word_id, z_mn = topic_id
						int topic_id = z_[m][n];
						--count_doc_topic_[m][topic_id];
						--count_topic_word_[topic_id][word_id];
						--sum_topic_word_[topic_id];
						//--sum_doc_word_[m];
						//sample
						z_[m][n] = sample(m, word_id);
						topic_id = z_[m][n];

						++count_doc_topic_[m][topic_id];
						++count_topic_word_[topic_id][word_id];
						++sum_topic_word_[topic_id];
						//++sum_doc_word_[m];
						++n;
					}
				}
			}
			std::cout << "sample iter #: " << iter_time << std::endl;
			
			// sample-in period
			if (iter_time > num_burn_in_ && iter_time % kSAMPLE_LAG == 0) {
				estimate();
				++sample_time;
			}
			if (sample_time == num_sample_) {
				average_param();
				break;
			}
			++iter_time;
		}
		return 0;
	}

	int sample(int m, int v) { // m : doc_id, k : topic_id, v : word_id
		//compute p(z_i = k | Z-i, W, alpha, beta).
		for (int k = 0; k < num_topic_; ++k) {
			if (k == 0) {
				p_[k] = (count_topic_word_[k][v] + beta_) / (sum_topic_word_[k] + sum_beta_) * 
					(count_doc_topic_[m][k] + alpha_);
			} else {
				p_[k] = p_[k-1] + (count_topic_word_[k][v] + beta_) / (sum_topic_word_[k] + sum_beta_) * 
					(count_doc_topic_[m][k] + alpha_);
			}
		}

		std::default_random_engine eng(::time(NULL));
		std::uniform_real_distribution<double> rng(0.0, 0.99999);
		double rand = rng(eng) * p_[num_topic_-1];

		int k = 0;
		for (k = 0; k < num_topic_; ++k) {
			if (p_[k] > rand) {
				break;
			}
		}
		return k;
	}

	int load(const std::string& filename) {
		std::ifstream infile(filename);
		if (!infile) {
			return -1;
		}

		std::string line;

		while (getline(infile, line)) {
			Document* doc = new Document();
			int num_term = doc->load(line);
			if (num_term <= 0) {
				std::cout << "load dot failed" << std::endl;
				continue;
			}
			corpus_.push_back(doc);
			num_word_ = num_word_ > num_term ? num_word_ : num_term;
		}
		//std::cout << num_topic_ << ' ' << num_word_ << std::endl;
		infile.close();
		return 0;
	}

	void estimate() {
		for (int m = 0; m < corpus_.size(); ++m) {
			for (int k = 0; k < num_topic_; ++k) {
				theta_[m][k] += (count_doc_topic_[m][k] + alpha_) / (sum_doc_word_[m] + sum_alpha_);
			}
		}
		for (int k = 0; k < num_topic_; ++k) {
			for (int v = 0; v < num_word_; ++v) {
				phi_[k][v] += (count_topic_word_[k][v] + beta_) / (sum_topic_word_[k] + sum_beta_);
			}
		}
	}

	void average_param() {
		for (int m = 0; m < corpus_.size(); ++m) {
			for (int k = 0; k < num_topic_; ++k) {
				theta_[m][k] /= num_sample_;
			}
		}
		for (int k = 0; k < num_topic_; ++k) {
			for (int v = 0; v < num_word_; ++v) {
				phi_[k][v] /= num_sample_;
			}
		}
	}

	void save() {
		std::ofstream outphi(model_name_+".phi");
		for (int i = 0; i < phi_.size(); ++i) {
			for (int j = 0; j < phi_[i].size(); ++j) {
				outphi << phi_[i][j] << " ";		
			}
			outphi << std::endl;
		}
		outphi.close();
	}

	void print_topic(std::string vocabname) {
		std::unordered_map<int, std::string> vocab;
		std::ifstream fin(vocabname);
		std::ofstream fout(model_name_ + ".topic");
		std::string term;
		int index = 0;
		while (getline(fin, term)) {
			vocab[index] = term;
			++index; 
		}
		for (int k = 0; k < num_topic_; ++k) {
			fout << "Topic " << k << " :" << std::endl;
			std::map<double, int> dict;
			for (int v = 0; v < num_word_; ++v) {
				dict[phi_[k][v]] = v;
			}
			std::map<double, int>::iterator it = dict.end();
			for (int i = 0; i < 25; ++i) {
				--it;
				fout << vocab[it->second] << std::endl;
			}
			fout << std::endl;
		}
	}

private:
	const int kSAMPLE_LAG = 10;
	//model setting
	int num_topic_;
	int num_burn_in_;
	int num_sample_;
	std::string model_name_;

	double alpha_, beta_; //

	//model parameter
	std::vector<VecInt> z_;
	std::vector<VecFloat> theta_; //theta[m][k] : doc m , topic k
	std::vector<VecFloat> phi_; // phi[k][v] : topic k, word v

	//count statistics
	std::vector<VecInt> count_doc_topic_;
	std::vector<VecInt> count_topic_word_;
	VecInt sum_doc_word_;
	VecInt sum_topic_word_;

	size_t num_word_; // size of vocab in corpus
	std::vector<Document* > corpus_;

	double sum_alpha_;
	double sum_beta_;

	VecFloat p_; // full conditional pro
};
