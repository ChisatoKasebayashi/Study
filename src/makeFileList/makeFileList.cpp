#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <vector>

using namespace std;
using namespace boost::filesystem;

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
    for(string& filename : filenames){
        cout << filename << endl;
    }

    return 0;
}
