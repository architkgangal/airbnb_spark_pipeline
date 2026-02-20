# Airbnb Spark Pipeline

An end-to-end **PySpark batch data pipeline** that processes Airbnb listings and reviews data to produce analytics-ready datasets.

This project demonstrates real-world data engineering patterns including data ingestion, cleaning, joins, aggregations, UDFs, Spark SQL, and batch output management.

---

## Problem Statement
Airbnb datasets are large, semi-structured, and span multiple entities (listings, hosts, reviews).  
The goal of this pipeline is to transform raw CSV data into meaningful, queryable datasets that answer common business and analytics questions.

---

## What This Pipeline Does
Using Apache Spark, the pipeline:

- Cleans and standardizes listing and review data
- Joins listings with reviews using listing identifiers
- Computes review volume per listing
- Calculates average prices by neighborhood and room type
- Identifies top hosts by number of listings
- Analyzes availability distribution
- Separates weekend vs weekday review activity
- Performs lightweight sentiment analysis on review text
- Generates basic data quality metrics

Each result is written as a separate output dataset.

---

## Tech Stack
- Python
- Apache Spark (PySpark)
- Spark SQL
- User Defined Functions (UDFs)
- Broadcast variables
- Git & GitHub

---

## Input Data
The pipeline expects the following input files:

- `listings.csv.gz`
- `reviews.csv.gz`

Data source: Inside Airbnb (public dataset)

> Raw data files are intentionally excluded from this repository.

---

## How to Run

```bash
spark-submit airbnb_spark_pipeline.py \
  --listings ./data/listings.csv.gz \
  --reviews  ./data/reviews.csv.gz \
  --output   ./data/output


python3.11 airbnb_spark_pipeline.py \
 --listings listings.csv \
 --reviews reviews.csv \
 --output ./output
