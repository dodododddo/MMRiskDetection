mongosh -u jrchen -p jrchen --authenticationDatabase jrchen << EOF
use jrchen
db.text.find().pretty()
EOF
