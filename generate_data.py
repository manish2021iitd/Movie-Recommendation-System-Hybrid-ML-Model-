"""
Generates a rich synthetic movie dataset for the recommendation system.
In production, replace with MovieLens or TMDB data.
"""
import pandas as pd
import numpy as np
import json
import os

np.random.seed(42)

MOVIES = [
    # Title, Year, Genres, Director, Cast (top 2), Description, Rating, Votes
    ("The Shawshank Redemption", 1994, ["Drama", "Crime"], "Frank Darabont",
     ["Tim Robbins", "Morgan Freeman"],
     "Two imprisoned men bond over years finding solace and eventual redemption through acts of common decency.",
     9.3, 2600000),
    ("The Godfather", 1972, ["Crime", "Drama"], "Francis Ford Coppola",
     ["Marlon Brando", "Al Pacino"],
     "The aging patriarch of an organized crime dynasty transfers control to his reluctant son.",
     9.2, 1900000),
    ("The Dark Knight", 2008, ["Action", "Crime", "Drama"], "Christopher Nolan",
     ["Christian Bale", "Heath Ledger"],
     "Batman raises the stakes in his war on crime with the Joker wreaking havoc on Gotham.",
     9.0, 2700000),
    ("Schindler's List", 1993, ["Biography", "Drama", "History"], "Steven Spielberg",
     ["Liam Neeson", "Ralph Fiennes"],
     "In German-occupied Poland, Oskar Schindler gradually becomes concerned for his Jewish workforce.",
     9.0, 1400000),
    ("Pulp Fiction", 1994, ["Crime", "Drama"], "Quentin Tarantino",
     ["John Travolta", "Uma Thurman"],
     "The lives of two mob hitmen, a boxer, a gangster and his wife intertwine in Los Angeles.",
     8.9, 2100000),
    ("Inception", 2010, ["Action", "Adventure", "Sci-Fi"], "Christopher Nolan",
     ["Leonardo DiCaprio", "Joseph Gordon-Levitt"],
     "A thief who steals corporate secrets through dream-sharing technology is given the inverse task.",
     8.8, 2400000),
    ("Interstellar", 2014, ["Adventure", "Drama", "Sci-Fi"], "Christopher Nolan",
     ["Matthew McConaughey", "Anne Hathaway"],
     "A team of explorers travel through a wormhole in space to ensure humanity's survival.",
     8.7, 2000000),
    ("The Matrix", 1999, ["Action", "Sci-Fi"], "Lana Wachowski",
     ["Keanu Reeves", "Laurence Fishburne"],
     "A computer programmer discovers that reality as he knows it is a simulation run by machines.",
     8.7, 1900000),
    ("Goodfellas", 1990, ["Biography", "Crime", "Drama"], "Martin Scorsese",
     ["Ray Liotta", "Robert De Niro"],
     "The story of Henry Hill and his life in the mob, covering his eventual downfall and testimony.",
     8.7, 1200000),
    ("Fight Club", 1999, ["Drama"], "David Fincher",
     ["Brad Pitt", "Edward Norton"],
     "An insomniac office worker and a soap salesman build a network of underground fight clubs.",
     8.8, 2200000),
    ("Forrest Gump", 1994, ["Drama", "Romance"], "Robert Zemeckis",
     ["Tom Hanks", "Robin Wright"],
     "The presidencies of Kennedy through Reagan unfold from the perspective of an Alabama man.",
     8.8, 2100000),
    ("The Silence of the Lambs", 1991, ["Crime", "Drama", "Thriller"], "Jonathan Demme",
     ["Jodie Foster", "Anthony Hopkins"],
     "A young FBI cadet must receive the help of an incarcerated cannibal killer to catch another serial killer.",
     8.6, 1500000),
    ("The Lord of the Rings: The Fellowship of the Ring", 2001, ["Adventure", "Drama", "Fantasy"], "Peter Jackson",
     ["Elijah Wood", "Ian McKellen"],
     "A meek hobbit and eight companions set out on a journey to destroy the powerful One Ring.",
     8.8, 1900000),
    ("The Lord of the Rings: The Return of the King", 2003, ["Adventure", "Drama", "Fantasy"], "Peter Jackson",
     ["Elijah Wood", "Viggo Mortensen"],
     "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo.",
     9.0, 1800000),
    ("Star Wars: A New Hope", 1977, ["Action", "Adventure", "Fantasy"], "George Lucas",
     ["Mark Hamill", "Harrison Ford"],
     "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids.",
     8.6, 1400000),
    ("Avengers: Endgame", 2019, ["Action", "Adventure", "Drama"], "Anthony Russo",
     ["Robert Downey Jr.", "Chris Evans"],
     "After the devastating events of Infinity War the universe is in ruins as the Avengers assemble once more.",
     8.4, 1200000),
    ("Parasite", 2019, ["Comedy", "Drama", "Thriller"], "Bong Joon-ho",
     ["Kang-ho Song", "Lee Sun-kyun"],
     "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Parks.",
     8.5, 800000),
    ("Spirited Away", 2001, ["Animation", "Adventure", "Family"], "Hayao Miyazaki",
     ["Daveigh Chase", "Suzanne Pleshette"],
     "A sullen ten-year-old girl wanders into a world ruled by gods and witches.",
     8.6, 750000),
    ("Whiplash", 2014, ["Drama", "Music"], "Damien Chazelle",
     ["Miles Teller", "J.K. Simmons"],
     "A promising young drummer enrolls at a cutthroat music conservatory where he falls under the wing of a demanding instructor.",
     8.5, 850000),
    ("La La Land", 2016, ["Comedy", "Drama", "Music"], "Damien Chazelle",
     ["Ryan Gosling", "Emma Stone"],
     "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations.",
     8.0, 700000),
    ("Mad Max: Fury Road", 2015, ["Action", "Adventure", "Sci-Fi"], "George Miller",
     ["Tom Hardy", "Charlize Theron"],
     "In a post-apocalyptic wasteland a woman rebels against a tyrannical ruler in search for her homeland.",
     8.1, 1000000),
    ("Get Out", 2017, ["Horror", "Mystery", "Thriller"], "Jordan Peele",
     ["Daniel Kaluuya", "Allison Williams"],
     "A young African-American visits his white girlfriend's family estate only to become ensnared in a more sinister real reason.",
     7.7, 700000),
    ("Knives Out", 2019, ["Comedy", "Crime", "Drama"], "Rian Johnson",
     ["Daniel Craig", "Chris Evans"],
     "A detective investigates the death of a crime novelist at a party for his 85th birthday.",
     7.9, 650000),
    ("The Grand Budapest Hotel", 2014, ["Adventure", "Comedy", "Crime"], "Wes Anderson",
     ["Ralph Fiennes", "Tony Revolori"],
     "A writer encounters the owner of an aging European hotel who tells him of his early years serving as a lobby boy.",
     8.1, 850000),
    ("Blade Runner 2049", 2017, ["Action", "Drama", "Mystery"], "Denis Villeneuve",
     ["Ryan Gosling", "Harrison Ford"],
     "Young Blade Runner K's discovery of a long-buried secret leads him to track down former Blade Runner Rick Deckard.",
     8.0, 550000),
    ("Arrival", 2016, ["Drama", "Mystery", "Sci-Fi"], "Denis Villeneuve",
     ["Amy Adams", "Jeremy Renner"],
     "A linguist works with the military to communicate with alien lifeforms after twelve mysterious spacecraft appear around the world.",
     7.9, 650000),
    ("Her", 2013, ["Drama", "Romance", "Sci-Fi"], "Spike Jonze",
     ["Joaquin Phoenix", "Scarlett Johansson"],
     "In a near future, a lonely writer develops an unlikely relationship with an operating system designed to meet his every need.",
     8.0, 700000),
    ("Eternal Sunshine of the Spotless Mind", 2004, ["Drama", "Romance", "Sci-Fi"], "Michel Gondry",
     ["Jim Carrey", "Kate Winslet"],
     "When their relationship turns sour a couple undergoes a medical procedure to have each other erased from their memories.",
     8.3, 950000),
    ("No Country for Old Men", 2007, ["Crime", "Drama", "Thriller"], "Joel Coen",
     ["Tommy Lee Jones", "Javier Bardem"],
     "Violence and mayhem ensue after a hunter stumbles upon a drug deal gone wrong and more than two million dollars in cash.",
     8.2, 1000000),
    ("There Will Be Blood", 2007, ["Drama"], "Paul Thomas Anderson",
     ["Daniel Day-Lewis", "Paul Dano"],
     "A story of family religion hatred and oil told through the life of a ruthless man.",
     8.2, 600000),
    ("Oldboy", 2003, ["Action", "Drama", "Mystery"], "Park Chan-wook",
     ["Choi Min-sik", "Yoo Ji-tae"],
     "After being kidnapped and imprisoned for fifteen years a man is released and searches for his captor in five days.",
     8.4, 600000),
    ("Cinema Paradiso", 1988, ["Drama", "Romance"], "Giuseppe Tornatore",
     ["Philippe Noiret", "Enzo Cannavale"],
     "A filmmaker recalls his childhood in a 1940s Sicilian village and his relationship with the local movie theater's projectionist.",
     8.5, 260000),
    ("Amélie", 2001, ["Comedy", "Romance"], "Jean-Pierre Jeunet",
     ["Audrey Tautou", "Mathieu Kassovitz"],
     "Amélie is an innocent and naive girl in Paris with her own sense of justice who decides to help those around her and along the way discovers love.",
     8.3, 750000),
    ("The Departed", 2006, ["Crime", "Drama", "Thriller"], "Martin Scorsese",
     ["Leonardo DiCaprio", "Matt Damon"],
     "An undercover cop and a mole in the police attempt to identify each other while infiltrating an Irish gang in South Boston.",
     8.5, 1300000),
    ("Zodiac", 2007, ["Crime", "Drama", "Mystery"], "David Fincher",
     ["Jake Gyllenhaal", "Mark Ruffalo"],
     "In the late 1960s the Zodiac Killer terrorizes the San Francisco Bay Area and a cartoonist becomes obsessed with finding the identity.",
     7.7, 500000),
    ("Se7en", 1995, ["Crime", "Drama", "Mystery"], "David Fincher",
     ["Brad Pitt", "Morgan Freeman"],
     "Two detectives a rookie and a veteran hunt a serial killer who uses the seven deadly sins as his motives.",
     8.6, 1600000),
    ("Heat", 1995, ["Action", "Crime", "Drama"], "Michael Mann",
     ["Al Pacino", "Robert De Niro"],
     "A group of professional bank robbers start to feel the heat from police when they unknowingly leave a clue at their latest heist.",
     8.3, 660000),
    ("Casino", 1995, ["Crime", "Drama"], "Martin Scorsese",
     ["Robert De Niro", "Sharon Stone"],
     "A tale of greed corruption and murder occur between two best friends a mafia enforcer and a casino executive.",
     8.2, 600000),
    ("2001: A Space Odyssey", 1968, ["Adventure", "Sci-Fi"], "Stanley Kubrick",
     ["Keir Dullea", "Gary Lockwood"],
     "After discovering a mysterious artifact buried beneath the Lunar surface mankind sets off on a quest to find its origins.",
     8.3, 700000),
    ("A Clockwork Orange", 1971, ["Crime", "Drama", "Sci-Fi"], "Stanley Kubrick",
     ["Malcolm McDowell", "Patrick Magee"],
     "In the future a sadistic gang leader is imprisoned and volunteers for a conduct-aversion experiment.",
     8.3, 850000),
    ("The Shining", 1980, ["Drama", "Horror"], "Stanley Kubrick",
     ["Jack Nicholson", "Shelley Duvall"],
     "A family heads to an isolated hotel for the winter where a sinister presence influences the father into violence.",
     8.4, 1000000),
    ("Full Metal Jacket", 1987, ["Drama", "War"], "Stanley Kubrick",
     ["Matthew Modine", "R. Lee Ermey"],
     "A pragmatic U.S. Marine observes the dehumanizing effects of the Vietnam War on his fellow recruits.",
     8.3, 700000),
    ("Apocalypse Now", 1979, ["Drama", "War"], "Francis Ford Coppola",
     ["Martin Sheen", "Marlon Brando"],
     "A U.S. Army officer serving in Vietnam is tasked with assassinating a renegade Special Forces Colonel.",
     8.4, 700000),
    ("Saving Private Ryan", 1998, ["Drama", "War"], "Steven Spielberg",
     ["Tom Hanks", "Tom Sizemore"],
     "Following the Normandy Landings a group of U.S. soldiers go behind enemy lines to retrieve a paratrooper whose brothers have been killed.",
     8.6, 1400000),
    ("Back to the Future", 1985, ["Adventure", "Comedy", "Sci-Fi"], "Robert Zemeckis",
     ["Michael J. Fox", "Christopher Lloyd"],
     "Marty McFly a 17-year-old is accidentally sent thirty years into the past in a time-traveling DeLorean.",
     8.5, 1200000),
    ("Jurassic Park", 1993, ["Action", "Adventure", "Sci-Fi"], "Steven Spielberg",
     ["Sam Neill", "Laura Dern"],
     "A pragmatic paleontologist touring an almost-complete theme park on an island in Central America is tasked with protecting a couple of kids.",
     8.2, 1000000),
    ("Titanic", 1997, ["Drama", "Romance"], "James Cameron",
     ["Leonardo DiCaprio", "Kate Winslet"],
     "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious ill-fated R.M.S. Titanic.",
     7.9, 1200000),
    ("Avatar", 2009, ["Action", "Adventure", "Fantasy"], "James Cameron",
     ["Sam Worthington", "Zoe Saldana"],
     "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world.",
     7.9, 1300000),
    ("The Truman Show", 1998, ["Comedy", "Drama", "Sci-Fi"], "Peter Weir",
     ["Jim Carrey", "Laura Linney"],
     "An insurance salesman discovers his whole life is actually a reality TV show.",
     8.2, 1100000),
    ("American Beauty", 1999, ["Drama", "Romance"], "Sam Mendes",
     ["Kevin Spacey", "Annette Bening"],
     "A sexually frustrated suburban father has a mid-life crisis after becoming infatuated with his daughter's best friend.",
     8.4, 1100000),
]

def generate_movies_df():
    records = []
    for i, m in enumerate(MOVIES):
        title, year, genres, director, cast, desc, rating, votes = m
        records.append({
            "movie_id": i + 1,
            "title": title,
            "year": year,
            "genres": "|".join(genres),
            "director": director,
            "cast": "|".join(cast),
            "description": desc,
            "imdb_rating": rating,
            "votes": votes,
            "poster_color": f"hsl({(i * 37) % 360}, 60%, 40%)"
        })
    return pd.DataFrame(records)


def generate_ratings_df(movies_df, n_users=500):
    """Generate synthetic user ratings with realistic patterns."""
    ratings = []
    n_movies = len(movies_df)

    # Create user preference profiles
    genre_list = ["Action", "Drama", "Crime", "Sci-Fi", "Comedy", "Romance",
                  "Thriller", "Horror", "Adventure", "Fantasy", "Biography",
                  "History", "Music", "Mystery", "War", "Animation"]

    user_profiles = []
    for u in range(n_users):
        # Each user prefers 2-4 genres
        n_pref = np.random.randint(2, 5)
        pref_genres = np.random.choice(genre_list, n_pref, replace=False)
        base_rating = np.random.uniform(3.0, 4.5)
        user_profiles.append({"genres": pref_genres, "base": base_rating})

    for u_idx, profile in enumerate(user_profiles):
        user_id = u_idx + 1
        # Rate 10-40 movies per user
        n_ratings = np.random.randint(10, 41)
        rated_movies = np.random.choice(n_movies, min(n_ratings, n_movies), replace=False)

        for m_idx in rated_movies:
            movie = movies_df.iloc[m_idx]
            movie_genres = set(movie["genres"].split("|"))
            user_pref_genres = set(profile["genres"])

            genre_match = len(movie_genres & user_pref_genres) / max(len(movie_genres), 1)
            imdb_factor = (movie["imdb_rating"] - 7.5) / 2.0

            base = profile["base"] + genre_match * 1.2 + imdb_factor * 0.8
            noise = np.random.normal(0, 0.4)
            rating = np.clip(base + noise, 1.0, 5.0)

            ratings.append({
                "user_id": user_id,
                "movie_id": int(movie["movie_id"]),
                "rating": round(rating * 2) / 2  # 0.5 increments
            })

    return pd.DataFrame(ratings)


if __name__ == "__main__":
    movies_df = generate_movies_df()
    ratings_df = generate_ratings_df(movies_df)

    out_dir = os.path.dirname(__file__)
    movies_df.to_csv(os.path.join(out_dir, "movies.csv"), index=False)
    ratings_df.to_csv(os.path.join(out_dir, "ratings.csv"), index=False)
    print(f"Generated {len(movies_df)} movies and {len(ratings_df)} ratings")
