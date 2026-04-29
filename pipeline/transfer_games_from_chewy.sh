# Transfer

# Transfer all lichess games from chewy to local machine

rsync -avP --partial --append-verify --checksum \
  "chewy:/home/landon/Projects/patzer/data/lichess_games/" \
  "data/lichess_games/"
