import praw
import torch
import sqlite3
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map

# Initialize FastAPI
app = FastAPI()

# Model Configuration (TinyLlama-1.1B-Chat)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model on CPU if CUDA is not available
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",  # Auto-detect CPU/GPU
        torch_dtype=torch.float16  # Use FP16 for GPU
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",  # Auto-detect CPU/GPU
        torch_dtype=torch.float32,  # Use FP32 for CPU
        quantization_config=None  # Disable quantization
    )

# Optimize with torch.compile
model = torch.compile(model, mode="reduce-overhead")

# SQLite Database (Avoid Reposting)
conn = sqlite3.connect("comments.db", check_same_thread=False)
c = conn.cursor()
c.execute("PRAGMA journal_mode=WAL;")  # Enable faster writes
c.execute("CREATE TABLE IF NOT EXISTS comments (post_id TEXT PRIMARY KEY)")
conn.commit()

# Reddit API Credentials (Read-Only Mode Enabled)
reddit = praw.Reddit(
    client_id="	Bitter-Bluebird8029",
    client_secret="9hSwO1K3q5dpYPk1UzLDcu7AwfNZ6A",
    username="Bitter-Bluebird8029",
    password="BabyDriver@098",
    user_agent="u/Bitter-Bluebird8029"
)

# Define Target Users
TARGET_USERS = ["user1", "user2", "user3"]

class CommentRequest(BaseModel):
    post_text: str

@app.post("/generate-comment")
def generate_comment_api(request: CommentRequest):
    """API endpoint to generate a comment based on post text."""
    try:
        comment = generate_comment(request.post_text)
        return {"comment": comment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_comment(post_text):
    """Generate a relevant comment using the optimized model."""
    input_text = f"Post: {post_text}\nResponse:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,  # Control only the number of new tokens generated
            do_sample=True,
            temperature=0.6,
            top_p=0.85
        )
    
    comment = tokenizer.decode(output[0], skip_special_tokens=True)
    return comment.split("Response:")[-1].strip()

@app.get("/monitor-reddit")
def monitor_reddit():
    """Monitor target users and reply to their posts."""
    responses = []
    batch_inserts = []

    for user in TARGET_USERS:
        try:
            redditor = reddit.redditor(user)
            for post in redditor.submissions.new(limit=3):
                if not c.execute("SELECT 1 FROM comments WHERE post_id=?", (post.id,)).fetchone():
                    comment_text = generate_comment(post.title + " " + post.selftext)
                    # post.reply(comment_text)  # Disabled to prevent actual commenting
                    print(f"Would comment on: {post.title}")
                    batch_inserts.append((post.id,))
                    responses.append({"post": post.title, "comment": comment_text})
                    time.sleep(10)  # Reduced wait time
        except Exception as e:
            print(f"Error processing user {user}: {e}")

    # Batch insert new post IDs to SQLite
    if batch_inserts:
        c.executemany("INSERT INTO comments VALUES (?)", batch_inserts)
        conn.commit()

    return {"status": "completed", "responses": responses}