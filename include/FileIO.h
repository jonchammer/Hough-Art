/* 
 * File:   FileIO.h
 * Author: Jon C. Hammer
 *
 * Created on September 19, 2016, 8:29 PM
 */

#ifndef FILEIO_H
#define FILEIO_H

#include <string>
#include <vector>
using std::string;
using std::vector;

// Platform-specific solutions from:
// http://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c

// Windows solution
#ifdef WIN32
#include <Windows.h>
vector<string> listFiles(const string& folder)
{
    vector<string> names;
    string search_path = folder + "/*.*";
    WIN32_FIND_DATA fd; 
    HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd); 
    if(hFind != INVALID_HANDLE_VALUE) { 
        do { 
            // read all (real) files in current folder
            // , delete '!' read other 2 default folder . and ..
            if(! (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ) {
                names.push_back(fd.cFileName);
            }
        }while(::FindNextFile(hFind, &fd)); 
        ::FindClose(hFind); 
    } 
    return names;
}

// Linux solution
#else
#include "dirent.h"
vector<string> listFiles(const string& folder)
{
    vector<string> names;
    DIR* dir = opendir(folder.c_str());
    
    if (dir != NULL)
    {
        dirent* ent;
        while ((ent = readdir(dir)) != NULL)
        {
            if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0)
                names.push_back(ent->d_name);
        }
        closedir(dir);
    }
    
    return names;
}
#endif
#endif /* FILEIO_H */

