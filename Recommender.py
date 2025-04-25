from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from DataWrangler import MovieData
import os
import pickle


class Recommender():
    
    def __init__(self):
        # Load preprocessed data and TF-IDF matrix from MovieData
        get_profiles_matrix = MovieData()

        self.recomandations = None
        self.movie_profiles = get_profiles_matrix.movie_profiles
        self.tf_idf_matrix = get_profiles_matrix.tfidf_matrix
        self.input_movie = None
        self.file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Model"))

    def find_input_movie(self):
        # Find the profile of the movie based on its title
        movie_profile = self.movie_profiles[self.movie_profiles["title"] == self.input_movie]
        self.input_movie = movie_profile


    def candidates_model(self):
        self.find_input_movie()
        print(self.input_movie)

        # Use Bag-of-Words columns to find similar movies
        bow_columns = self.movie_profiles.columns[15:]
        input_vector = self.input_movie[bow_columns].values
        similar_movies = self.movie_profiles[self.movie_profiles["movieId"] != self.input_movie["movieId"].values[0]]

        # Cosine similarity on BoW features
        sims = cosine_similarity(input_vector, similar_movies[bow_columns].values)[0]
        similar_movies = similar_movies.copy()
        similar_movies["similarity"] = sims        

        # First filter: top 60 by similarity, then top 30 by mean rating
        similar_movies = similar_movies.sort_values("similarity", ascending=False).head(60)  
        candidates = similar_movies.sort_values("mean_rating", ascending=False).head(30)

        return candidates

    
    def scoring(self):
        candidates = self.candidates_model()
        
        # Use TF-IDF similarity between input and candidates
        input_tfidf = self.tf_idf_matrix[self.input_movie.index[0]]
        candidates_tfidf = self.tf_idf_matrix[candidates.index]

        sims = cosine_similarity(input_tfidf, candidates_tfidf)[0]
        candidates = candidates.copy()
        candidates["similarity"] = sims

        # Return top 15 based on TF-IDF similarity
        scoring_set = candidates.sort_values("similarity", ascending=False).head(15)

        return scoring_set
        

    def pred_engagement_level(self):
        # Features used to predict engagement
        feature_list = ["mean_time", "std_time", "last_engagement", 
                        "total_engagement", "release_year_unix", 
                        "since_release_to_mean", "engagement_span"]
        
        model_file = os.path.join(self.file_path, "pred_engagement_model.pkl")

        # Load model if exists, otherwise train a new one
        if os.path.exists(model_file):
            with open(model_file, "rb") as file:
                random_forest = pickle.load(file)
        else:            
            X_train = self.movie_profiles[feature_list]
            y_train = self.movie_profiles["engagement_level"]
            random_forest = RandomForestClassifier(n_estimators=10)
            random_forest.fit(X_train, y_train)

            with open(model_file, "wb") as file:
                pickle.dump(random_forest, file)

        return random_forest, feature_list


    def recommander(self):
        scoring_set = self.scoring()
        model, feature_list = self.pred_engagement_level()

        # Predict engagement level for top candidates
        pred = model.predict(scoring_set[feature_list])
        scoring_set["pred_engagement_level"] = pred
       
        # Order by engagement level from Very High to Very Low
        engagement_order = ["Very High", "High", "Medium", "Low", "Very Low"]
        scoring_set = scoring_set[scoring_set["pred_engagement_level"].isin(engagement_order)].copy()
        scoring_set["engagement_rank"] = scoring_set["pred_engagement_level"].apply(lambda x: engagement_order.index(x))
        scoring_set = scoring_set.sort_values("pred_engagement_level")

        # Select top 3 high engagement + 2 recent releases
        high_engamnate = scoring_set.sort_values(by="engagement_rank", ascending=True).head(3)
        scoring_set.drop(high_engamnate.index, inplace=True)
        recent_realesed = scoring_set.sort_values(by="year", ascending=False).head(2)

        recommendations = pd.concat([recent_realesed, high_engamnate])
        recommendations.reset_index(drop=True, inplace=True)

        return recommendations
