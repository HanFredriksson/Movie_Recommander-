Project Description

This project is a content-based movie recommender system that suggests similar movies using genres, tags, ratings, and engagement data.
It works in three stages, filtering and narrowing down the selection based on different features and models.

The dataset comes from the MovieLens data set and combines the movies, tags, and ratings data.
All of this is handled in the DataWrangler class, where I build a profile for each movie and also create a TF-IDF matrix from tags and genres.


Methods
--------

The feature engineering part is about merging all the useful info from the datasets — mainly tags, ratings, and movie data.
From that, I calculate a few features: average rating, total number of ratings, and a weighted mean.
The weighted mean helps give a more balanced rating by considering both score and how many people rated the movie.

For engagement, features are made from at when tags and ratings were made, using timestamp data.
From that features are buuilt : mean_time, std_time, last_engagement, total_engagement, release_year_unix, since_release_to_mean, engagement_span.
Engagement_span: is from first to last post. 
Since_release_to_mean: is time to the mean time from release. Giving an gauage of the movie is still talk about or not.
The idea is to get a better picture of how much attention the movie has gotten over time.
Since the datasets were pretty big, this was also a way to compress the data and focus on just movie-level features.

Then I use KMeans clustering to sort movies into rough engagement levels: from Very Low to Very High.
These clusters help train a Random Forest model that predicts engagement later on.

Genres are turned into a Bag-of-Words vector. This is later used to find similar movies using cosine similarity.

TF-IDF is used on both tags and genres to try to capture the more unique or important features for each movie.
That vector gets used with cosine similarity too, to find better matches beyond just genre.

In the final step, I use a Random Forest Classifier to predict the engagement level for the top 10 candidate movies — and then filter it down to 5 solid recommendations.

Each stage applies filters and re-ranking based on the scores: similarity, predicted engagement, and weighted rating.


Assumptions / Limitations
--------------------------

This recommender only uses content-based features, no collaborative filtering.
Even though the tags and ratings come from users, the system doesn't model individual user behavior.
This makes it easier to use all available data without needing personalized history.

It only works for movies already in the dataset.
It also assumes the genres and tags are accurate and describe the movie well.

The app is built to run in the terminal. It's a bit simple, but it lets you test and compare movie suggestions easily.
Collaborative filtering could have given more personal results, but the goal here was to keep things simple and focus on content.

The three-step recommendation was chosen to add variation and deeper filtering — not just picking movies with the same genre.
TF-IDF and cosine similarity help find more subtle connections between movies.


Design Choices
--------------

The biggest choice was to go fully content-based so I could use more of the dataset.
The tags dataset is huge, and trying to model user preferences on top of that would’ve made things messy fast.

Instead, I built profiles around the movies — merging all info (tags, genres, ratings) into one compact representation.
This also helped movies with fewer tags still get more detailed features thanks to their genres.

I picked Random Forest in the final step to predict engagement. It works well with non-linear patterns and handles noise — which is useful since not all tags and genres are guaranteed to be great descriptions of a movie.

The app is built with two main classes: one to wrangle the data and one for the actual recommender.
The terminal interface is kept basic — the main focus was on building the recommender and getting the data wrangling working well.

The Movie Profiles, tf-idf matrix and models are saved to file. This is to be able to run the Recommender faster and not need to run all the data and models everytime.