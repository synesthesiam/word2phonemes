LANGUAGES := en
BASE_DICT_FILES := data/en.dict
CLEAN_DICT_FILES := data/en.csv

all: $(CLEAN_DICT_FILES)

%.csv: %.dict
	bin/clean-dictionary.py < $< > $@


