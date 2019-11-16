#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

int main(int argc, char** argv){
    if(argc != 2){
        cerr << "Usage:" << argv[0] << " dirname" << endl;
        return EXIT_FAILURE;
    }
    path dir(argv[1]);
    if(!exists(dir) || !is_directory(dir)){
        cerr << dir << " : Not such directory" << endl;
        return EXIT_FAILURE;
    }
    vector<string> filenames;
    try{
        directory_iterator end;
        for(directory_iterator it(dir); it != end; ++it){
            if(!is_directory(it->path())){
                filenames.push_back(it->path().filename().string());
            }
        }
    }catch(const filesystem_error& e){
        cerr << e.what() << endl;
    }


    Mat res_img;
    string img_dir;
    string folder = argv[1];
    for(string& filename : filenames){
        img_dir =  folder + filename;
        // 画像をグレイスケールで読み込む-> リサイズする-> 保存する
        cout << img_dir << endl;
        Mat src_img = imread(img_dir, 0);
        resize(src_img, res_img, Size(), 0.0625, 0.0625);
        //imshow("train_data", res_img);
        imwrite("./"+filename,res_img);
    }

    return 0;
}
