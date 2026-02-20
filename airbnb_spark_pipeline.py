import argparse
import re
from datetime import datetime

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import BooleanType, StringType, IntegerType

# ----------------------------
# Args
# ----------------------------
parser = argparse.ArgumentParser(description="Airbnb PySpark Pipeline (from your bootcamp exercises)")
parser.add_argument("--listings", required=True, help="Path to listings dataset (csv/csv.gz)")
parser.add_argument("--reviews", required=True, help="Path to reviews dataset (csv/csv.gz)")
parser.add_argument("--output", required=True, help="Output directory (base folder)")
args = parser.parse_args()

# ----------------------------
# Spark session
# ----------------------------
spark = (
    SparkSession.builder
    .appName("Airbnb End-to-End PySpark Pipeline")
    .getOrCreate()
)

# ----------------------------
# Read
# ----------------------------
read_opts = dict(
    header=True,
    inferSchema=True,
    sep=",",
    quote='"',
    escape='"',
    multiLine=True,
    mode="PERMISSIVE"
)

listings_raw = spark.read.csv(args.listings, **read_opts)
reviews_raw  = spark.read.csv(args.reviews,  **read_opts)

# listings.csv.gz :selecting columns that are actually used
listings = (
    listings_raw.select(
        "id", "name", "price",
        "host_id", "host_name",
        "neighbourhood_cleansed", "room_type",
        "availability_365"
    )
    .withColumn("price_num", F.regexp_replace(F.col("price").cast("string"), "[$,]", "").cast("double"))
)

# reviews.csv.gz : selecting columns that are actually used
reviews = reviews_raw.select("listing_id", "id", "date", "comments")


# ----------------------------
# 1) Reviews per listing
# ----------------------------
l = listings.alias("l")
r = reviews.alias("r")

listings_reviews = l.join(r, F.col("l.id") == F.col("r.listing_id"), "inner")

reviews_per_listing = (
    listings_reviews
    .groupBy(F.col("l.id").alias("listing_id"), F.col("l.name").alias("listing_name"))
    .agg(F.count(F.col("r.id")).alias("reviews_count"))
    .orderBy(F.col("reviews_count").desc())
)

reviews_per_listing.write.mode("overwrite").option("header", True).csv(f"{args.output}/01_reviews_per_listing")


# ----------------------------
# 2) Spark SQL version
# ----------------------------
listings.createOrReplaceTempView("listings")
reviews.createOrReplaceTempView("reviews")

sql_reviews_per_listing = spark.sql("""
SELECT
    l.id   AS listing_id,
    l.name AS listing_name,
    COUNT(r.id) AS reviews_count
FROM listings l
INNER JOIN reviews r
    ON l.id = r.listing_id
GROUP BY l.id, l.name
ORDER BY reviews_count DESC
""")

sql_reviews_per_listing.write.mode("overwrite").option("header", True).csv(f"{args.output}/02_sql_reviews_per_listing")

# ----------------------------
# 3) Price category UDF
# ----------------------------
@F.udf(StringType())
def price_cat(price_num):
    if price_num is None:
        return "Unknown"
    if price_num < 50:
        return "Budget"
    if 50 <= price_num < 150:
        return "Mid-range"
    return "Luxury"

price_category_counts = (
    listings
    .filter(F.col("price_num").isNotNull())
    .withColumn("price_category", price_cat(F.col("price_num")))
    .groupBy("price_category")
    .agg(F.count("*").alias("listings_count"))
    .orderBy(F.col("listings_count").desc())
)

price_category_counts.write.mode("overwrite").option("header", True).csv(f"{args.output}/03_price_category_counts")

# ----------------------------
# 4) Avg price by neighbourhood + room_type
# ----------------------------
avg_price_by_area_room = (
    listings
    .filter(F.col("price_num").isNotNull())
    .groupBy("neighbourhood_cleansed", "room_type")
    .agg(
        F.avg("price_num").alias("avg_price"),
        F.count("*").alias("listings_count")
    )
    .orderBy(F.col("avg_price").desc())
)

avg_price_by_area_room.write.mode("overwrite").option("header", True).csv(f"{args.output}/04_avg_price_by_neighborhood_room_type")

# ----------------------------
# 5) Weekend UDF
# ----------------------------
invalid_dates = spark.sparkContext.accumulator(0)

@F.udf(BooleanType())
def is_weekend(date_str):
    global invalid_dates
    try:
        if date_str is None:
            invalid_dates += 1
            return False
        d = datetime.strptime(date_str, "%Y-%m-%d")  # matches your reviews.csv.gz date format
        return d.weekday() >= 5
    except Exception:
        invalid_dates += 1
        return False

weekend_review_counts = (
    reviews
    .withColumn("is_weekend", is_weekend(F.col("date")))
    .groupBy("listing_id")
    .agg(
        F.sum(F.col("is_weekend").cast("int")).alias("weekend_reviews"),
        F.count("*").alias("total_reviews")
    )
    .orderBy(F.col("weekend_reviews").desc())
)

weekend_review_counts.write.mode("overwrite").option("header", True).csv(f"{args.output}/05_weekend_review_counts")

# ----------------------------
# 6) Sentiment -> avg sentiment per listing
# ----------------------------
positive_words = {
    "good","great","excellent","amazing","fantastic","wonderful","pleasant",
    "lovely","nice","enjoyed","superb","clean","comfortable"
}
negative_words = {
    "bad","terrible","awful","horrible","disappointing","poor","hate",
    "unpleasant","dirty","noisy","worst"
}

pos_b = spark.sparkContext.broadcast(positive_words)
neg_b = spark.sparkContext.broadcast(negative_words)

def sentiment_score(comment):
    if comment is None:
        return 0
    words = re.findall(r"[a-z']+", comment.lower())
    pos = sum(1 for w in words if w in pos_b.value)
    neg = sum(1 for w in words if w in neg_b.value)
    return pos - neg

sentiment_udf = F.udf(sentiment_score, IntegerType())

reviews_with_sentiment = reviews.withColumn("sentiment_score", sentiment_udf(F.col("comments")))

l = listings.alias("l")
r = reviews_with_sentiment.alias("r")

avg_sentiment_per_listing = (r
    .join(l.select(F.col("id").alias("l_listing_id"), F.col("name").alias("listing_name")),
          F.col("r.listing_id") == F.col("l_listing_id"),
          "inner")
    .groupBy(F.col("l_listing_id").alias("listing_id"), F.col("listing_name"))
    .agg(
        F.avg(F.col("r.sentiment_score")).alias("avg_sentiment"),
        F.count("*").alias("reviews_count")
    )
    .orderBy(F.col("avg_sentiment").desc())
)

avg_sentiment_per_listing.write.mode("overwrite").option("header", True).csv(f"{args.output}/06_avg_sentiment_per_listing")


# ----------------------------
# 7) Avg comment length
# ----------------------------
reviews_with_comment_len = reviews.withColumn("comment_length", F.length(F.col("comments")))

l = listings.alias("l")
r = reviews_with_comment_len.alias("r")

avg_comment_length = (
    r
    .join(l.select(F.col("id").alias("l_listing_id"), F.col("name").alias("listing_name")),
          F.col("r.listing_id") == F.col("l_listing_id"),
          "inner")
    .groupBy(F.col("l_listing_id").alias("listing_id"), F.col("listing_name"))
    .agg(
        F.avg(F.col("r.comment_length")).alias("avg_comment_length"),
        F.count("*").alias("reviews_count")
    )
    .filter(F.col("reviews_count") >= 5)
    .orderBy(F.col("avg_comment_length").desc())
)

avg_comment_length.write.mode("overwrite").option("header", True).csv(f"{args.output}/07_avg_comment_length_min5reviews")


# ----------------------------
# 8) Data quality summary
# ----------------------------
dq_listings = spark.createDataFrame(
    [
        ("listings_rows", listings.count()),
        ("listings_null_price", listings.filter(F.col("price_num").isNull()).count()),
        ("listings_null_name", listings.filter(F.col("name").isNull()).count()),
        ("distinct_hosts", listings.select("host_id").dropDuplicates().count()), 
    ],
    ["metric", "value"]
)

dq_reviews = spark.createDataFrame(
    [
        ("reviews_rows", reviews.count()),
        ("reviews_null_date", reviews.filter(F.col("date").isNull()).count()),
        ("reviews_null_comments", reviews.filter(F.col("comments").isNull()).count()),
        ("invalid_dates_seen_by_udf", invalid_dates.value),
    ],
    ["metric", "value"]
)

dq_listings.unionByName(dq_reviews).write.mode("overwrite").option("header", True).csv(f"{args.output}/08_data_quality_report")

spark.stop()