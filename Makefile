LANGUAGES := en fr
BASE_DICT_FILES := $(foreach lang,$(LANGUAGES),data/$(lang)/dictionary.dict)
CLEAN_DICT_FILES := $(foreach lang,$(LANGUAGES),data/$(lang)/dictionary.csv)

all: $(CLEAN_DICT_FILES)

%.csv: %.dict
	bin/clean-dictionary.py < $< > $@

