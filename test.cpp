#include <fstream> // Include this to use ofstream
#include <iostream>
#include <string>

int main() {
  std::ofstream file;              // Create an ofstream object
  const char* filename = "filename.txt";
  file.open(filename);       // Open a file (creates it if it doesn’t exist)
  file << "Stuff to write. adding smtg new";        // Write to the file
  file.close();
}
