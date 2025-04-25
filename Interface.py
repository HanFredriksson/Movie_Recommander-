from Recommender import Recommender 
import os 

# Function to clear the terminal screen for a clean user interface
def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

# Main function to run the interactive recommender system
def run_recommender():

    get_recommendations = Recommender()


    while True:
        clear_terminal()  
        input_title = input("Pick a movie: ")  
        
        get_recommendations.input_movie = input_title

        try:
            # Generate recommendations
            recommendations = get_recommendations.recommander()
        except Exception as e:
            # If movie is not found or another error occurs, notify the user
            print(f"\nCould not find or process movie '{input_title}'. Error: {e}")
            input("\nPress Enter to try again...")
            continue  

        clear_terminal() 

        # Display the movie that the user selected
        print(get_recommendations.input_movie)
        selected = get_recommendations.input_movie.iloc[0] 
        print(f"You picked the movie: {selected['title']}, From {selected['year']}, Genres: {selected['genres']}\n")

        print("----------------------------")

        # Print the recommended movies
        for _, movie in recommendations.iterrows():
            print(f"{movie['title']}, From: {movie['year']}, Genres: {movie['genres']}")
        
        print("----------------------------\n")

       
        again = input("Do you want to pick another movie? (y/n): ").lower()
        if again != "y":
            print("Goodbye!")
            break  


if __name__ == "__main__":
    run_recommender()
