FROM erikvl87/languagetool:6.7

USER root

RUN apk add fasttext fasttext-libs

RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

USER languagetool