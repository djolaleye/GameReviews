# GameReviews

This project attempts to forecast the amount of expected revenue a video game can generate based on its price, the number of reviews, and its given rating on Metacritic and RAWG websites. 

Features: log reviews, log price, Metacritic rating, RAWG rating  
Weights: 1.85, 0.64, -0.02, 0.01  

The website ratings are essentially irrelevant towards the final prediction. The reviews feature is overly prominent, as it is the only one with a strongy linear correlation with revenue.  

This multiple linear regression model has strong explanatory power regarding video game revenue. 
- R^2 training = 0.999, testing = 0.999
- 75% of game revenue predicted within 6% of the target
- Lower revenue targets were associated with higher model error

This model does not accomplish its original intent, as the predictive utility is low due to heavily relying on total reviews. This is not helpful business-wise, as reviews are more of a byproduct of success rather than a controllable variable to predict success. Finding more preemptive metrics to use would provide more actionable benefit. 

Some potential features for future projects: hours to beat (content), wishlists, trailer views, website/social media interactions
Better insights could also be drawn by focusing on games of different genres, release windows and/or game platforms.
