# AutoRespondReddit

### This code is a Python script that integrates several technologies to monitor Reddit posts from specific users, generate comments using a language model, and store data in an SQLite database to avoid reposting. Below is a step-by-step explanation of the code:

1- Uses FastAPI to expose an API for generating comments.

2- Monitors Reddit posts from specific users.

3- Generates comments using a pre-trained language model.

4- Stores post IDs in an SQLite database to avoid reposting.

5- Avoids actual commenting on Reddit (commented out for safety).

### Implementation of code

praw: A Python library for interacting with the Reddit API.

torch: PyTorch, used for deep learning tasks.

sqlite3: A lightweight database for storing post IDs to avoid reposting.

time: Used for adding delays between actions.

FastAPI: A modern web framework for building APIs.

pydantic: A library for data validation and settings management.

transformers: Hugging Face's library for working with pre-trained language models.

accelerate: A library for optimizing model inference across devices.
```python
import praw
import torch
import sqlite3
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map
```
Creates a FastAPI application instance to handle HTTP requests.
Specifies the pre-trained language model (TinyLlama-1.1B-Chat) to use for generating comments.
Loads the tokenizer for the model.
```python
app = FastAPI()

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```
Loads the model on a GPU if available (using FP16 for efficiency) or on a CPU (using FP32).

device_map="auto" ensures the model is loaded on the appropriate device (CPU/GPU).
```python
if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float32,
        quantization_config=None
    )
```
Compiles the model using PyTorch's torch.compile to optimize performance.
Connects to an SQLite database (comments.db) to store post IDs.
Creates a table comments if it doesn't already exist, with post_id as the primary key.
Enables Write-Ahead Logging (WAL) for faster database writes.
```python
model = torch.compile(model, mode="reduce-overhead")

conn = sqlite3.connect("comments.db", check_same_thread=False)
c = conn.cursor()
c.execute("PRAGMA journal_mode=WAL;")
c.execute("CREATE TABLE IF NOT EXISTS comments (post_id TEXT PRIMARY KEY)")
conn.commit()
```
Initializes a Reddit API client using praw with the provided credentials.
The user_agent identifies the script to Reddit.
```python
reddit = praw.Reddit(
    client_id="Bitter-Bluebird8029",
    client_secret="9hSwO1K3q5dpYPk1UzLDcu7AwfNZ6A",
    username="Bitter-Bluebird8029",
    password="BabyDriver@098",
    user_agent="u/Bitter-Bluebird8029"
)
```
Specifies the list of Reddit users whose posts will be monitored.
Defines a Pydantic model for validating incoming API requests. It expects a post_text field.
```python
TARGET_USERS = ["user1", "user2", "user3"]

class CommentRequest(BaseModel):
    post_text: str
```
Defines a POST endpoint /generate-comment that accepts a post_text input and returns a generated comment.
Calls the generate_comment function to generate the comment and handles errors using HTTPException.
```python
@app.post("/generate-comment")
def generate_comment_api(request: CommentRequest):
    try:
        comment = generate_comment(request.post_text)
        return {"comment": comment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
Takes post_text as input and formats it for the model.
Tokenizes the input and moves it to the appropriate device (CPU/GPU).
Generates a comment using the model with the following parameters:
max_new_tokens=60: Limits the length of the generated comment.
do_sample=True: Enables sampling for more diverse outputs.
temperature=0.6: Controls randomness (lower values make outputs more deterministic).
top_p=0.85: Implements nucleus sampling for better quality.
Decodes the generated tokens into text and extracts the comment.
```python
def generate_comment(post_text):
    input_text = f"Post: {post_text}\nResponse:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.6,
            top_p=0.85
        )
    
    comment = tokenizer.decode(output[0], skip_special_tokens=True)
    return comment.split("Response:")[-1].strip()
```
Monitors new posts from the target users.

For each post, checks if it has already been commented on by querying the SQLite database.
If not, generates a comment using the generate_comment function.
Stores the post ID in the database to avoid reposting.
Adds a delay of 10 seconds between actions to avoid rate limits.
Returns a summary of the posts and generated comments.
```python
@app.get("/monitor-reddit")
def monitor_reddit():
    responses = []
    batch_inserts = []

    for user in TARGET_USERS:
        try:
            redditor = reddit.redditor(user)
            for post in redditor.submissions.new(limit=3):
                if not c.execute("SELECT 1 FROM comments WHERE post_id=?", (post.id,)).fetchone():
                    comment_text = generate_comment(post.title + " " + post.selftext)
                    print(f"Would comment on: {post.title}")
                    batch_inserts.append((post.id,))
                    responses.append({"post": post.title, "comment": comment_text})
                    time.sleep(10)
        except Exception as e:
            print(f"Error processing user {user}: {e}")

    if batch_inserts:
        c.executemany("INSERT INTO comments VALUES (?)", batch_inserts)
        conn.commit()

    return {"status": "completed", "responses": responses}
```
