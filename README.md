# Game Revenue
Video games continue to be one of the most popular forms of entertainment. However, publishers and new game studios often grapple with the uncertainty of how profitable their product will become. This project attempts to take some of the subjectivity out of the question and forecast the amount of expected revenue a video game can generate based on its sale price, number of reviews, and given rating on Metacritic and RAWG websites. 


## Data Sourcing
The data used was gathered from the RAWG api and Steam via steamspy and Game Data Crunch. 

## Model Building
Features: log reviews, log price, Metacritic rating, RAWG rating  
Weights: 1.85, 0.64, -0.02, 0.01  

After scaling, the website ratings surprisingly end up being essentially irrelevant towards the final prediction. The reviews feature is overly prominent, as it is the only predictor with a strong linear correlation with revenue.  

This multiple linear regression model seems to show strong explanatory power towards game revenue. 
- R^2 training = 0.999, testing = 0.999
- 75% of game revenue is able to be predicted within 6% of the target
- Lower revenue targets were associated with higher model error

## Conclusion
This model does not accomplish its original intent, as the predictive utility is low due to heavily relying on post-launch factors like total reviews and critic scores. This is not helpful business-wise, as these are more of a byproduct of success rather than controllable variables for preemptuvely predicting success. A linear regression also doesn't properly model the relationship between the chosen predictors and financial success. 

A future iteration of this project would be better served using potential features like: hours to beat (content), wishlists, trailer views, and website/social media interactions. I would also be interested in a future analysis using an unsupervised method instead to observe commonalities in games that reached different levels of success. Better insights could also be drawn by focusing on specific genres, release windows, and game platforms.


<img src="https://github.com/djolaleye/GameReviews/blob/main/plots/review_rev.png" width=300 align=left>
<img src="https://github.com/djolaleye/GameReviews/blob/main/plots/price_rev.png" width=300 align=center>
<img src="https://github.com/djolaleye/GameReviews/blob/main/plots/rawg_rev.png" width=300 align=left>  
<img src="https://github.com/djolaleye/GameReviews/blob/main/plots/meta_rev.png" width=300 align=center>  
