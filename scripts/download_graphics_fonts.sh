INSALLATION_DIR="data/static/fonts/roboto_mono"

LINK="https://fonts.google.com/download?family=Roboto%20Mono"

mkdir -p $INSALLATION_DIR

wget $LINK -O $INSALLATION_DIR/roboto_mono.zip
unzip $INSALLATION_DIR/roboto_mono.zip -d $INSALLATION_DIR
rm $INSALLATION_DIR/roboto_mono.zip
