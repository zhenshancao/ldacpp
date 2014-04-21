#include "lda.h"

#include <iostream>
#include <string>
#include <fstream>

using namespace std;

int main() {
	string filename = "trn.dat";
	string vocab = "vocat.txt";
	LDA* model = new LDA();
	model->load(filename);
	model->init();
	model->train();
	model->save();
	model->print_topic(vocab);
}
