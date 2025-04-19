import bcrypt
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://hamza:hamza123@cluster0.bulzdjl.mongodb.net/Binance?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["Binance"]
collection = db["users"]

username = "testuser"
password = "test123"

hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

collection.insert_one({
    "username": username,
    "password": hashed_pw
})

print("✅ Test user added successfully!")
