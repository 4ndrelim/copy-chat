import os
import sys
import sqlite3
import logging
import time
import json
import requests
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

TARGET_USERNAME = "naval"
FETCH_REPLIES = True
FETCH_CONVERSATION_TWEETS = True
FETCH_TWEETS = False
MAX_TWEETS_PER_USER = 50_000
TWEETS_PER_PAGE = 100
DB_FILE = "tweets.db"
API_CALL_DELAY = 2
TWITTER_API_BASE_URL = "https://api.x.com/2"
TWEET_FIELDS = [
    "id", "text", "created_at", "author_id", "conversation_id",
    "in_reply_to_user_id", "public_metrics", "entities", "referenced_tweets",
    "context_annotations", "source", "lang", "possibly_sensitive",
    "edit_history_tweet_ids"
]
USER_FIELDS = ["id", "name", "username", "created_at", "description",
               "verified", "public_metrics", "profile_image_url", "location"]
EXPANSIONS = [
    "author_id",
    "referenced_tweets.id",
    "entities.mentions.username",
    "in_reply_to_user_id",
    "referenced_tweets.id.author_id",
    "attachments.media_keys"
]
MEDIA_FIELDS = [
    "media_key", "type", "url", "preview_image_url"
]
START_TIME = "2023-01-01T00:00:00Z"

# for auth access
_BEARER_TOKEN = None


def setup_database() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute('''
    CREATE TABLE IF NOT EXISTS tweets (
        id TEXT PRIMARY KEY,
        author_id TEXT,
        text TEXT,
        created_at TEXT,
        conversation_id TEXT,
        in_reply_to_user_id TEXT,
        like_count INTEGER,
        reply_count INTEGER,
        retweet_count INTEGER,
        quote_count INTEGER,
        view_count INTEGER,
        is_reply BOOLEAN,
        is_thread_continuation BOOLEAN,
        in_thread_position INTEGER,
        thread_id TEXT,
        tweet_source TEXT,
        language TEXT,
        has_media BOOLEAN,
        media_types TEXT,
        url_count INTEGER,
        hashtag_count INTEGER,
        mention_count INTEGER,
        hour_of_day INTEGER,
        day_of_week INTEGER,
        json_data TEXT,
        fetched_at TEXT
    )
    ''')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS thread_info (
        thread_id TEXT,
        tweet_id TEXT,
        position INTEGER,
        PRIMARY KEY (thread_id, tweet_id),
        FOREIGN KEY (tweet_id) REFERENCES tweets(id)
    )
    ''')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS pagination_state (
        author_id TEXT,
        username TEXT,
        fetch_replies BOOLEAN,
        last_tweet_id TEXT,
        last_fetch_time TEXT,
        next_token TEXT,
        PRIMARY KEY (author_id, fetch_replies)
    )
    ''')

    conn.commit()

    return conn

def get_pagination_state(conn: sqlite3.Connection, author_id: str, username: str, fetch_replies: bool) -> Dict[str, Any]:
    cursor = conn.cursor()

    cursor.execute(
        "SELECT last_tweet_id, last_fetch_time, next_token FROM pagination_state WHERE author_id = ? AND fetch_replies = ?",
        (author_id, fetch_replies)
    )

    row = cursor.fetchone()

    if row:
        return {
            "last_tweet_id": row[0],
            "last_fetch_time": row[1],
            "next_token": row[2]
        }

    return {
        "last_tweet_id": None,
        "last_fetch_time": None,
        "next_token": None
    }

def update_pagination_state(
    conn: sqlite3.Connection,
    author_id: str,
    username: str,
    fetch_replies: bool,
    last_tweet_id: Optional[str] = None,
    next_token: Optional[str] = None
) -> None:
    cursor = conn.cursor()

    now = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO pagination_state (author_id, username, fetch_replies, last_tweet_id, last_fetch_time, next_token)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (author_id, fetch_replies)
        DO UPDATE SET
            username = ?,
            last_tweet_id = COALESCE(?, last_tweet_id),
            last_fetch_time = ?,
            next_token = ?
        """,
        (
            author_id, username.lower(), fetch_replies, last_tweet_id, now, next_token,
            username.lower(), last_tweet_id, now, next_token
        )
    )

    conn.commit()


def get_bearer_token() -> str:
    load_dotenv()

    api_key = os.getenv("TWITTER_API_KEY")
    api_secret = os.getenv("TWITTER_API_KEY_SECRET")

    if not api_key or not api_secret:
        raise ValueError("TWITTER_API_KEY and TWITTER_API_KEY_SECRET must be set in .env file")

    auth_header = f"{api_key}:{api_secret}"
    encoded_auth = base64.b64encode(auth_header.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}

    logger.info("Authenticating with Twitter API...")
    try:
        response = requests.post(
            "https://api.x.com/oauth2/token",
            headers=headers,
            data=data
        )

        if response.status_code != 200:
            error_message = f"Twitter API authentication failed: status={response.status_code}, body={response.text}"
            logger.error(error_message)
            raise Exception(error_message)

        token_response = response.json()

        if token_response.get("token_type", "").lower() != "bearer":
            error_message = f"Unexpected token type: {token_response.get('token_type')}"
            logger.error(error_message)
            raise Exception(error_message)

        bearer_token = token_response.get("access_token")
        if not bearer_token:
            raise Exception("No access token in response")

        logger.info("Successfully obtained Twitter API bearer token")
        return bearer_token

    except Exception as e:
        logger.error(f"Error getting bearer token: {str(e)}")
        raise

def get_auth_headers() -> Dict[str, str]:
    global _BEARER_TOKEN

    try:
        if _BEARER_TOKEN:
            return {
                "Authorization": f"Bearer {_BEARER_TOKEN}",
                "Content-Type": "application/json"
            }

        _BEARER_TOKEN = get_bearer_token()

    except Exception as e:
        logger.warning(f"Failed to get token programmatically: {str(e)}")
        logger.warning("Falling back to token from environment variables")

        load_dotenv()
        _BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
        if not _BEARER_TOKEN:
            raise ValueError("TWITTER_BEARER_TOKEN not found in environment variables and programmatic retrieval failed")

    return {
        "Authorization": f"Bearer {_BEARER_TOKEN}",
        "Content-Type": "application/json"
    }

# def get_oauth_auth() -> Dict[str, str]:
#     load_dotenv()

#     api_key = os.getenv("TWITTER_API_KEY")
#     api_key_secret = os.getenv("TWITTER_API_KEY_SECRET")
#     access_token = os.getenv("TWITTER_ACCESS_TOKEN")
#     access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")

#     if not all([api_key, api_key_secret, access_token, access_token_secret]):
#         raise ValueError("Twitter OAuth credentials not found in environment variables")

#     return {
#         "api_key": api_key,
#         "api_key_secret": api_key_secret,
#         "access_token": access_token,
#         "access_token_secret": access_token_secret
#     }

def handle_rate_limit(response: requests.Response) -> bool:
    if response.status_code == 429:
        reset_time = int(response.headers.get("x-rate-limit-reset", 0))
        current_time = int(time.time())

        wait_time = max(reset_time - current_time, 0) + 5  # Add 5 seconds buffer

        if wait_time > 0:
            logger.info(f"Rate limited. Waiting for {wait_time} seconds...")
            time.sleep(wait_time)
            return True

    return False

def make_twitter_request(url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    headers = get_auth_headers()

    response = requests.get(url, headers=headers, params=params)

    # twitter has weird limits
    if handle_rate_limit(response):
        response = requests.get(url, headers=headers, params=params)

    if response.status_code != 200:
        error_message = f"Twitter API request failed: {response.status_code} - {response.text}"
        logger.error(error_message)

        # invalid pagination token error
        if response.status_code == 400 and "Invalid 'next_token'" in response.text:
            logger.warning("Invalid pagination token detected. Resetting pagination state.")
            raise ValueError("Invalid pagination token")

        raise Exception(error_message)

    return response.json()

def get_user_id(username: str) -> str:
    try:
        username = username.lstrip('@')
        url = f"{TWITTER_API_BASE_URL}/users/by/username/{username}"

        response = make_twitter_request(url)

        if not response.get("data"):
            raise ValueError(f"User @{username} not found.")

        return response["data"]["id"]

    except Exception as e:
        logger.error(f"Error getting user ID for @{username}: {str(e)}")
        raise

def check_existing_tweets(conn: sqlite3.Connection, author_id: str) -> Dict[str, bool]:
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM tweets WHERE author_id = ?", (author_id,))
    existing_tweets = {row[0]: True for row in cursor.fetchall()}
    logger.info(f"Found {len(existing_tweets)} existing tweets for author_id {author_id}")
    return existing_tweets

def fetch_tweets(
    fetch_replies: bool,
    pagination_token: Optional[str] = None,
    max_results: int = TWEETS_PER_PAGE,
    until_id: Optional[str] = None
) -> Dict[str, Any]:
    try:
        username = TARGET_USERNAME

        # we don't care about retweets in our project
        if fetch_replies:
            query = f"from:{username} is:reply -is:retweet"
        else:
            query = f"from:{username} -is:reply -is:retweet"

        url = f"{TWITTER_API_BASE_URL}/tweets/search/all"

        params = {
            "query": query,
            "max_results": max_results,
            "tweet.fields": ",".join(TWEET_FIELDS),
            "user.fields": ",".join(USER_FIELDS),
            "media.fields": ",".join(MEDIA_FIELDS),
            "expansions": ",".join(EXPANSIONS),
            "sort_order": "recency",
            "start_time": START_TIME
        }

        if pagination_token:
            params["next_token"] = pagination_token
        if until_id:
            params["until_id"] = until_id

        logger.info(f"making /tweets/search/all request with query: {query}")
        response = make_twitter_request(url, params)

        return {
            "tweets": response.get("data", []),
            "includes": response.get("includes", {}),
            "meta": response.get("meta", {})
        }

    except Exception as e:
        logger.error(f"error fetching {'replies' if fetch_replies else 'tweets'}: {str(e)}")
        raise

def fetch_tweets_by_ids(tweet_ids: List[str]) -> Dict[str, Any]:
    if not tweet_ids:
        return {"tweets": [], "includes": {}}

    batch_size = 100
    all_tweets = []
    all_includes = {"users": [], "media": [], "tweets": []}

    for i in range(0, len(tweet_ids), batch_size):
        batch = tweet_ids[i:i + batch_size]

        try:
            url = f"{TWITTER_API_BASE_URL}/tweets"
            ids_param = ",".join(batch)
            params = {
                "ids": ids_param,
                "tweet.fields": ",".join(TWEET_FIELDS),
                "user.fields": ",".join(USER_FIELDS),
                "media.fields": ",".join(MEDIA_FIELDS),
                "expansions": ",".join(EXPANSIONS),
            }

            logger.info(f"Fetching batch of {len(batch)} tweets by ID")
            response = make_twitter_request(url, params)

            tweets = response.get("data", [])
            includes = response.get("includes", {})

            all_tweets.extend(tweets)

            logger.info(f"first tweet timestamp: {tweets[0]['created_at']}")
            logger.info(f"last tweet timestamp: {tweets[-1]['created_at']}")

            for key, values in includes.items():
                if key in all_includes:
                    all_includes[key].extend(values)
                else:
                    all_includes[key] = values

            if API_CALL_DELAY > 0 and i + batch_size < len(tweet_ids):
                logger.info(f"sleeping for {API_CALL_DELAY} seconds...")
                time.sleep(API_CALL_DELAY)

        except Exception as e:
            logger.error(f"error fetching tweets by IDs: {str(e)}")
            continue

    return {"tweets": all_tweets, "includes": all_includes}

def get_conversation_tweet_ids(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    cursor = conn.cursor()

    cursor.execute("""
        SELECT DISTINCT conversation_id
        FROM tweets
        WHERE is_reply = 1
    """)

    conversation_ids = [row[0] for row in cursor.fetchall() if row[0]]

    conversation_tweet_map = {}

    for conv_id in conversation_ids:
        cursor.execute("""
            SELECT json_data
            FROM tweets
            WHERE conversation_id = ?
        """, (conv_id,))

        tweet_jsons = [row[0] for row in cursor.fetchall() if row[0]]
        referenced_ids = set()

        for json_str in tweet_jsons:
            try:
                tweet_data = json.loads(json_str)
                refs = tweet_data.get("referenced_tweets", [])
                for ref in refs:
                    if ref.get("id") and ref.get("type") in ["replied_to", "quoted"]:
                        referenced_ids.add(ref["id"])
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error processing tweet JSON: {str(e)}")
                continue

        if referenced_ids:
            conversation_tweet_map[conv_id] = list(referenced_ids)

    return conversation_tweet_map

def get_all_referenced_tweet_ids(conn: sqlite3.Connection) -> List[str]:
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM tweets")
    existing_ids = {row[0] for row in cursor.fetchall()}

    cursor.execute("SELECT json_data FROM tweets WHERE is_reply = 1")

    referenced_ids = set()
    for row in cursor.fetchall():
        if not row[0]:
            continue

        try:
            tweet_data = json.loads(row[0])
            refs = tweet_data.get("referenced_tweets", [])
            for ref in refs:
                if ref.get("id") and ref.get("type") in ["replied_to", "quoted"]:
                    referenced_ids.add(ref["id"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing tweet JSON: {str(e)}")
            continue

    return list(referenced_ids - existing_ids)

# uses up a lot of quota
def fetch_conversation_by_id(conversation_id: str) -> Dict[str, Any]:
    try:
        url = f"{TWITTER_API_BASE_URL}/tweets/search/all"

        query = f"conversation_id:{conversation_id}"

        params = {
            "query": query,
            "max_results": 100,
            "tweet.fields": ",".join(TWEET_FIELDS),
            "user.fields": ",".join(USER_FIELDS),
            "media.fields": ",".join(MEDIA_FIELDS),
            "expansions": ",".join(EXPANSIONS),
            "sort_order": "recency",
        }

        logger.info(f"Fetching conversation with ID: {conversation_id}")
        response = make_twitter_request(url, params)

        return {
            "tweets": response.get("data", []),
            "includes": response.get("includes", {}),
            "meta": response.get("meta", {})
        }

    except Exception as e:
        logger.error(f"Error fetching conversation ID {conversation_id}: {str(e)}")
        return {"tweets": [], "includes": {}, "meta": {}}

def fetch_conversation_threads(conn: sqlite3.Connection) -> int:
    logger.info("fetching full conversation threads...")

    cursor = conn.cursor()
    cursor.execute("SELECT id FROM tweets")
    existing_tweets = {row[0]: True for row in cursor.fetchall()}
    logger.info(f"Found {len(existing_tweets)} existing tweets in the database")

    cursor.execute("""
        SELECT DISTINCT conversation_id
        FROM tweets
        WHERE is_reply = 1
    """)

    conversation_ids = [row[0] for row in cursor.fetchall() if row[0]]
    logger.info(f"Found {len(conversation_ids)} conversations to fetch")

    total_new_tweets = 0

    for i, conversation_id in enumerate(conversation_ids):
        logger.info(f"Processing conversation {i+1}/{len(conversation_ids)} (ID: {conversation_id})")

        try:
            result = fetch_conversation_by_id(conversation_id)
            tweets = result.get("tweets", [])
            includes = result.get("includes", {})

            if not tweets:
                logger.info(f"No tweets found for conversation ID: {conversation_id}")
                if API_CALL_DELAY > 0:
                    logger.info(f"Sleeping for {API_CALL_DELAY} seconds...")
                    time.sleep(API_CALL_DELAY)
                continue

            logger.info(f"Fetched {len(tweets)} tweets in conversation ID: {conversation_id}")

            tweet_dates = [tweet.get("created_at") for tweet in tweets if tweet.get("created_at")]
            if tweet_dates:
                earliest_date = min(tweet_dates)
                latest_date = max(tweet_dates)
                logger.info(f"Conversation date range: {earliest_date} to {latest_date}")

            new_tweets_count = save_tweets_to_db(conn, tweets, existing_tweets, includes)
            logger.info(f"Saved {new_tweets_count} new tweets from conversation to database")

            total_new_tweets += new_tweets_count

            if i < len(conversation_ids) - 1 and API_CALL_DELAY > 0:
                logger.info(f"Sleeping for {API_CALL_DELAY} seconds...")
                time.sleep(API_CALL_DELAY)

        except Exception as e:
            logger.error(f"error processing conversation ID {conversation_id}: {str(e)}")
            continue

    logger.info(f"Completed fetching conversation threads. Added {total_new_tweets} new tweets to database")
    return total_new_tweets


def extract_thread_information(tweets: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    thread_mapping = {}

    for tweet in tweets:
        conv_id = tweet.get("conversation_id")
        if not conv_id:
            continue

        if conv_id not in thread_mapping:
            thread_mapping[conv_id] = []

        thread_mapping[conv_id].append(tweet)

    for thread_id, thread_tweets in thread_mapping.items():
        thread_mapping[thread_id] = sorted(
            thread_tweets,
            key=lambda t: t.get("created_at", "")
        )

    return thread_mapping

def save_tweets_to_db(conn: sqlite3.Connection, tweets: List[Dict[str, Any]], existing_tweets: Dict[str, bool], includes: Dict[str, Any] = None) -> int:
    cursor = conn.cursor()
    now = datetime.now().isoformat()

    thread_mapping = extract_thread_information(tweets)

    new_tweets_count = 0
    for tweet in tweets:
        tweet_id = tweet.get("id")
        if not tweet_id:
            continue

        if tweet_id in existing_tweets:
            logger.debug(f"skipping existing tweet: {tweet_id}")
            continue

        new_tweets_count += 1
        existing_tweets[tweet_id] = True

        author_id = tweet.get("author_id", "")
        text = tweet.get("text", "")
        created_at = tweet.get("created_at", "")
        conversation_id = tweet.get("conversation_id", "")
        in_reply_to_user_id = tweet.get("in_reply_to_user_id", "")

        metrics = tweet.get("public_metrics", {})
        like_count = metrics.get("like_count", 0)
        reply_count = metrics.get("reply_count", 0)
        retweet_count = metrics.get("retweet_count", 0)
        quote_count = metrics.get("quote_count", 0)
        view_count = metrics.get("impression_count", 0)
        referenced_tweets = tweet.get("referenced_tweets", [])
        is_reply = any(ref.get("type") == "replied_to" for ref in referenced_tweets)

        is_thread_continuation = False
        in_thread_position = 0
        thread_id = conversation_id

        if conversation_id and conversation_id in thread_mapping:
            thread_tweets = thread_mapping[conversation_id]
            if len(thread_tweets) > 1:
                for i, t in enumerate(thread_tweets):
                    if t.get("id") == tweet_id:
                        in_thread_position = i
                        is_thread_continuation = i > 0
                        break

        entities = tweet.get("entities", {})
        urls = entities.get("urls", [])
        hashtags = entities.get("hashtags", [])
        mentions = entities.get("mentions", [])

        url_count = len(urls)
        hashtag_count = len(hashtags)
        mention_count = len(mentions)

        has_media = False
        media_types = []

        if "attachments" in tweet and "media_keys" in tweet["attachments"]:
            has_media = True
            media_keys = tweet["attachments"]["media_keys"]

            tweet_includes = {}
            if "includes" in tweet:
                tweet_includes = tweet["includes"]
            elif includes is not None:
                tweet_includes = includes

            includes_media = tweet_includes.get("media", [])
            for media in includes_media:
                if "media_key" in media and media.get("media_key") in media_keys:
                    media_types.append(media.get("type", "unknown"))

        hour_of_day = None
        day_of_week = None

        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                hour_of_day = dt.hour
                day_of_week = dt.weekday()
            except (ValueError, TypeError):
                pass

        json_data = json.dumps(tweet)

        cursor.execute(
            """
            INSERT OR REPLACE INTO tweets (
                id, author_id, text, created_at, conversation_id,
                in_reply_to_user_id, like_count, reply_count,
                retweet_count, quote_count, view_count, is_reply,
                is_thread_continuation, in_thread_position, thread_id,
                tweet_source, language, has_media, media_types,
                url_count, hashtag_count, mention_count,
                hour_of_day, day_of_week, json_data, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tweet_id, author_id, text, created_at, conversation_id,
                in_reply_to_user_id, like_count, reply_count,
                retweet_count, quote_count, view_count, is_reply,
                is_thread_continuation, in_thread_position, thread_id,
                tweet.get("source"), tweet.get("lang"),
                has_media, ",".join(media_types) if media_types else None,
                url_count, hashtag_count, mention_count,
                hour_of_day, day_of_week, json_data, now
            )
        )

        if thread_id and in_thread_position is not None:
            cursor.execute(
                """
                INSERT OR REPLACE INTO thread_info (thread_id, tweet_id, position)
                VALUES (?, ?, ?)
                """,
                (thread_id, tweet_id, in_thread_position)
            )

    conn.commit()
    return new_tweets_count

def get_latest_tweet_id(conn: sqlite3.Connection, author_id: str, fetch_replies: bool) -> Optional[str]:
    cursor = conn.cursor()
    query = """
    SELECT id FROM tweets
    WHERE author_id = ? AND is_reply = ?
    ORDER BY created_at DESC
    LIMIT 1
    """
    cursor.execute(query, (author_id, fetch_replies))
    result = cursor.fetchone()
    return result[0] if result else None

def main() -> None:
    logger.info(f"Starting Twitter scraper for @{TARGET_USERNAME} {'replies' if FETCH_REPLIES else 'tweets'}")

    conn = setup_database()

    try:
        if FETCH_TWEETS:
            # if we hit the same user it doesn't count as quota
            user_id = get_user_id(TARGET_USERNAME)
            logger.info(f"Found user ID for @{TARGET_USERNAME}: {user_id}")

            existing_tweets = check_existing_tweets(conn, user_id)

            latest_tweet_id = get_latest_tweet_id(conn, user_id, FETCH_REPLIES)
            if latest_tweet_id:
                logger.info(f"Found latest tweet ID: {latest_tweet_id}")

            pagination_state = get_pagination_state(conn, user_id, TARGET_USERNAME, FETCH_REPLIES)
            next_token = pagination_state["next_token"]

            if next_token:
                logger.info(f"Continuing from pagination token: {next_token}")
            else:
                logger.info("Starting from the beginning (no pagination token)")

            total_tweets = 0
            new_tweets = 0
            page_count = 0
            no_new_tweets_count = 0

            while total_tweets < MAX_TWEETS_PER_USER:
                page_count += 1
                logger.info(f"Fetching page {page_count}...")

                try:
                    result = fetch_tweets(
                        fetch_replies=FETCH_REPLIES,
                        pagination_token=next_token,
                        max_results=min(TWEETS_PER_PAGE, MAX_TWEETS_PER_USER - total_tweets),
                        until_id=None  # need to be set to None for date filtering
                    )
                except ValueError as e:
                    if "Invalid pagination token" in str(e):
                        logger.info("Resetting pagination token and starting from the beginning")
                        next_token = None
                        update_pagination_state(
                            conn=conn,
                            author_id=user_id,
                            username=TARGET_USERNAME,
                            fetch_replies=FETCH_REPLIES,
                            next_token=None
                        )
                        continue
                    else:
                        raise

                tweets = result.get("tweets", [])
                includes = result.get("includes", {})

                if not tweets:
                    logger.info("No more tweets returned from the API.")
                    break

                if tweets:
                    tweet_dates = [tweet.get("created_at") for tweet in tweets if tweet.get("created_at")]
                    if tweet_dates:
                        earliest_date = min(tweet_dates)
                        latest_date = max(tweet_dates)
                        logger.info(f"Date range of this batch: {earliest_date} to {latest_date}")

                logger.info(f"Fetched {len(tweets)} tweets.")
                new_tweets_in_batch = save_tweets_to_db(conn, tweets, existing_tweets, includes)
                logger.info(f"Saved {new_tweets_in_batch} new tweets to database.")

                # TODO: should check for all no new tweets
                if new_tweets_in_batch == 0:
                    no_new_tweets_count += 1
                    logger.info(f"No new tweets in this batch. Consecutive pages with no new tweets: {no_new_tweets_count}")

                    if no_new_tweets_count >= 2:
                        logger.info("Multiple pages with no new tweets. Assuming we've reached previously fetched data.")
                        break
                else:
                    no_new_tweets_count = 0

                total_tweets += len(tweets)
                new_tweets += new_tweets_in_batch
                logger.info(f"Total tweets fetched so far: {total_tweets} of maximum {MAX_TWEETS_PER_USER}")
                logger.info(f"New tweets added so far: {new_tweets}")

                meta = result.get("meta", {})
                next_token = meta.get("next_token")

                if not next_token:
                    next_token = meta.get("pagination_token")

                if tweets:
                    last_tweet_id = tweets[-1]["id"]
                    update_pagination_state(
                        conn=conn,
                        author_id=user_id,
                        username=TARGET_USERNAME,
                        fetch_replies=FETCH_REPLIES,
                        last_tweet_id=last_tweet_id,
                        next_token=next_token
                    )

                if not next_token:
                    logger.info("No more pages available.")
                    break

                if total_tweets >= MAX_TWEETS_PER_USER:
                    logger.info(f"Reached maximum number of tweets ({MAX_TWEETS_PER_USER})")
                    break

                if API_CALL_DELAY > 0:
                    logger.info(f"Sleeping for {API_CALL_DELAY} seconds...")
                    time.sleep(API_CALL_DELAY)

            logger.info(f"Twitter scraper completed. Total tweets fetched: {total_tweets}, new tweets added: {new_tweets}")

        if FETCH_CONVERSATION_TWEETS:
            logger.info("Now fetching conversation data...")
            conversation_tweets_added = fetch_conversation_threads(conn)
            logger.info(f"Added {conversation_tweets_added} tweets from conversations")

    except Exception as e:
        logger.error(f"Error running Twitter scraper: {str(e)}")
        raise

    finally:
        conn.close()

if __name__ == "__main__":
    main()
