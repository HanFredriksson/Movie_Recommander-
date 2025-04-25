import pandas as pd
import os 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse

class MovieData():
    
    def __init__(self):
        # Initialize variables and paths
        self.movie_profiles = None
        self.tfidf_matrix = None

        self.data_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        self.movie_profiles_path = os.path.join(self.data_path, "Data", "movie_profiles.csv")
        self.tfidf_matrix_path = os.path.join(self.data_path, "Data", "tfidf_matrix.npz")

        # Load or create profiles
        if os.path.exists(self.movie_profiles_path) and os.path.exists(self.tfidf_matrix_path):
            self.load_profiles()
        else:
            self.make_movie_profiles()
            self.save_profiles()


    def save_profiles(self):
        # Save processed movie profiles and tf-idf matrix
        self.movie_profiles.to_csv(self.movie_profiles_path, index=False)
        sparse.save_npz(self.tfidf_matrix_path, self.tfidf_matrix)


    def load_profiles(self):
        # Load movie profiles and tf-idf matrix from file
        self.movie_profiles = pd.read_csv(self.movie_profiles_path)
        self.tfidf_matrix = sparse.load_npz(self.tfidf_matrix_path)


    def make_movie_profiles(self):
        # Load raw datasets
        movies_data = pd.read_csv(os.path.join(self.data_path, "..", "Data", "movies.csv"))
        ratings_data = pd.read_csv(os.path.join(self.data_path,"..", "Data", "ratings.csv"))
        tags_data = pd.read_csv(os.path.join(self.data_path, "..", "Data", "tags.csv"))

        # Extract year from title and clean title
        movies_data.insert(2, "year",  movies_data["title"].str.extract(r"\((\d{4})\)"))
        movies_data["title"] = movies_data["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)  

        # Create features
        ratings_features = self.rating_aggrigator(ratings_data)
        time_engagment_features = self.time_features(ratings_data, tags_data, movies_data)
        bow_vector = self.bow_vectorizer(movies_data)

        # Merge features into movie dataset
        feature_list = [ratings_features, time_engagment_features, bow_vector]
        for feature in feature_list:
            movies_data = movies_data.merge(feature, on="movieId", how="left")
        
        # Filter movies that have tags and drop missing data
        tags_data.dropna(inplace=True)
        valid_movie_ids = tags_data["movieId"].unique()
        movies_data = movies_data[movies_data["movieId"].isin(valid_movie_ids)].copy()
        movies_data.dropna(inplace=True)
        movies_data.reset_index(drop=True, inplace=True)

        # Create TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer(tags_data, movies_data)
        self.movie_profiles = movies_data

    
    def tfidf_input(self, tags_data, movies_data):
        # Prepare combined input of genres + tags for TF-IDF
        tags_data["tag"] = tags_data["tag"].astype(str) 
        tfidf_input = tags_data.groupby(["movieId"], as_index=False).aggregate({"tag" : " ".join})
        tfidf_input["tag"] = tfidf_input["tag"].str.replace(" ", "|")

        tfidf_input = tfidf_input.merge(movies_data[["movieId", "genres"]], on="movieId", how="left")
        tfidf_input["tag"] = tfidf_input["tag"].fillna("")    
        tfidf_input["tf_idf_input"] = tfidf_input["genres"] + " " + tfidf_input["tag"]

        tfidf_input = tfidf_input.drop(["tag", "genres"], axis=1)
        tfidf_input.dropna(inplace=True)

        # Ensure same order as movie_profiles
        tfidf_input = tfidf_input[tfidf_input["movieId"].isin(movies_data["movieId"])]
        tfidf_input = tfidf_input.set_index("movieId").loc[movies_data["movieId"]].reset_index()

        return tfidf_input
    

    def tfidf_vectorizer(self, tags_data, movies_data):
        # Apply TF-IDF on genres + tags
        tfidf_input = self.tfidf_input(tags_data, movies_data)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(tfidf_input["tf_idf_input"])
        return tfidf_matrix


    def rating_aggrigator(self, rating_data):
        # Compute average rating and total number of ratings per movie
        feature_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        rating_count = rating_data.groupby("movieId")["rating"].value_counts().rename("count").to_frame()
        rating_count.reset_index(level=["rating"], inplace=True)

        rating_pivot = rating_count.pivot(columns="rating", values="count").reset_index().fillna(0)
        
        existing_cols = [col for col in feature_list if col in rating_pivot.columns]
        weighted_sum = sum(float(r) * rating_pivot[r] for r in existing_cols)
        total_ratings = rating_pivot[existing_cols].sum(axis=1)

        rating_pivot["mean_rating"] = weighted_sum / total_ratings
        rating_pivot["total_ratings"] = total_ratings
        rating_pivot.drop(feature_list, axis=1, inplace=True)

        return rating_pivot


    def time_features(self, rating_data, tag_data, movies_data):
        # Create time-based features from tag and rating timestamps
        time_tags = tag_data.drop(["userId", "tag"],axis=1)
        time_rating = rating_data.drop(["userId", "rating"], axis=1)
        time_data = pd.concat([time_tags, time_rating])

        time_data_features = time_data.groupby("movieId")["timestamp"].agg(["mean", "std", "min", "max", "count"]).reset_index()
        time_data_features.columns = ["movieId", "mean_time", "std_time", "min_time", "last_engagement", "total_engagement"]

        # Calculate time-based engagement features
        time_data_features["release_date"] = pd.to_datetime(movies_data["year"], format="%Y")
        time_data_features["release_year_unix"] = (time_data_features["release_date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        time_data_features["since_release_to_mean"] = time_data_features["mean_time"] - time_data_features["release_year_unix"]
        time_data_features["engagement_span"] = time_data_features["last_engagement"] - time_data_features["min_time"]
        time_data_features.drop(["release_date", "min_time"], axis=1, inplace=True)

        # Normalize and classify engagement
        feature_list = ["mean_time", "std_time", "last_engagement", "total_engagement", "release_year_unix", "since_release_to_mean", "engagement_span"]
        scaler = MinMaxScaler()
        time_data_features[feature_list] = scaler.fit_transform(time_data_features[feature_list])
        time_data_features.dropna(inplace=True)

        time_data_features = self.engagement_classifier(time_data_features, feature_list)

        return time_data_features


    def bow_vectorizer(self, movies_data):
        # Create a Bag-of-Words vector from genre strings
        genres = ["Action", "Adventure", "Animation", "Children", "Comedy", 
                  "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", 
                  "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", 
                  "War", "Western"]
        
        vectorizer = CountVectorizer()
        vectorizer.fit(genres)

        genre_matrix = vectorizer.transform(movies_data["genres"]).toarray()
        bow_input = pd.DataFrame(genre_matrix, columns=vectorizer.get_feature_names_out())
        bow_input["movieId"] = movies_data["movieId"].values

        return bow_input


    def engagement_classifier(self, time_data_features, feature_list):
        # Cluster engagement into 5 levels using KMeans
        engagement_cluster = KMeans(n_clusters=5)
        engagement_cluster.fit(time_data_features[feature_list])
        time_data_features["clusters"] =  engagement_cluster.labels_

        # Rank clusters by std_time to assign levels
        cluster_means = time_data_features.groupby("clusters")[feature_list].mean()
        sorted_clusters = cluster_means.sort_values(by="std_time").index

        labels = ["Very Low", "Low", "Medium", "High", "Very High"]
        class_labels = {cluster: labels[i] for i, cluster in enumerate(sorted_clusters)}
        time_data_features["engagement_level"] = time_data_features["clusters"].map(class_labels)
        time_data_features.drop("clusters", axis=1, inplace=True)

        return time_data_features
