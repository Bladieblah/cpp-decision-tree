PROGNAME = tree.out
SRCDIR = src/decision_tree_c/
OBJDIR = obj/
INCDIR = include/

SRC	= $(SRCDIR)decision_tree.cpp $(SRCDIR)csr_matrix.cpp

CC = g++ -std=c++17

WARNFLAGS = -Wall -Wno-deprecated-declarations -Wno-writable-strings
CFLAGS = -g -O3 $(WARNFLAGS) -MD -Iinclude/ -I/usr/local/include
LDFLAGS = -L/usr/local/lib

# Do some substitution to get a list of .o files from the given .cpp files.
OBJFILES = $(patsubst $(SRCDIR)%.cpp, $(OBJDIR)%.o, $(SRC))
INCFILES = $(patsubst $(SRCDIR)%.cpp, $(INCDIR)%.hpp, $(SRC))

.PHONY: all clean install test

all: $(PROGNAME)

$(PROGNAME): $(OBJFILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)%.o: $(SRCDIR)%.cpp
	$(CC) -c $(CFLAGS) -o $@ $<

clean:
	rm -fv $(PROGNAME) $(OBJFILES)
	rm -fv $(OBJDIR)*.d

install:
	rm -rf build/
	pip install .

test:
	python -m unittest -v

-include $(OBJFILES:.o=.d)
