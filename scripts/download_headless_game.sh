INSALLATION_DIR="data/game"

LINK="http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip"
ARCHIVE_PASSPHRASE="iagreetotheeula"

wget $LINK -P $INSALLATION_DIR
unzip -P $ARCHIVE_PASSPHRASE $INSALLATION_DIR/StarCraftII.zip -d $INSALLATION_DIR
rm $INSALLATION_DIR/StarCraftII.zip
mv StarCraftII/* .
