Hi! Hope you'll have fun and enjoy PINA as much as I did, good luck with everything. I thought I'd leave a few tips on PINA classes and similar stuff that popped up in the past three months, in the hope you'll be able to far surpass my results and prove that it is indeed possible to get good approximations! :3

To start things off I've tried to put a few comments on what we're doing so as to assist you in going through my files, I commented them in the same order I created them so I suggest you look at them in the order aliev-panfilov --> 1dequation --> monodomain for the simple fact that a few files (like the tuner which has the same identical structure) of the first two folders have more comments that were not repeated in the monodomain folder.

- learning rates: I did a few tunes and saw that usually the best results are with a lr between 1e-4 and 1e-3, with 1e-4 being themost consistent throughout cases. This is because higher lrs (1e-3, 5e-3) proved to give less stable losses if you look up TensorBoard logs, while lower lrs (1e-5, 5e-5) often gave less reliable results and higher final losses. In any case I'll leave you a few files on the earlier trials on the first two models (as for the last model the three different Condition settings are all in separate files here, you can start from there adjusting the parameters/tuning), I'll give you the links at the end of this message. I know the description of each image is very short, if you need help understanding feel free to email me at matteo.bertocchi6@studio.unibo.it (my SISSA account already got deactivated :_) )

- epochs: in the 1d and monodomain cases the epochs are the same as the thesis, while in the A-P model I tried varying them to see how the results behaved. The best thing to use is TensorBoard, sorry for not leaving logs but you can easily get them yourself with just a single tune setting epochs as an item in the config dict.

- custom geometries, conditions and sampling: if you wish to create custom geometries, especially for the monodomain equation, I highly recommend looking at tutorial6. I think CartesianDomain and EllipsoidDomain should be sufficient. Usually you need these custom geometries for the conditions of your model, in which you'll use Condition. Look at the __init__.py file in the pina.condition folder to see what pairs can be used inside Condition, but 99% of times you'll be using either domain-equation (and here you'll need your geometries) or input-target (I used it both in the aliev-panfilov and monodomain models, these are usually the ones you use to implement data). After implementing your conditions you'll need to discretise all the domains for the domain-equation conditions, the two most common discretisations are grid and random. Outside of CartesianDomain the only sampling option available is random, that's what you'll have to use for circles in the monodomain and similar stuff. CartesianDomain also can be passed a dict in sample_mode to further specify how to sample along each coordinate (for example how large to make the grid).

I hope these additional remarks and the aforementioned comments will be enough for you to get a hold of my previous tries, another useful resource might be my previous messages with Pierfrancesco since I often attached the results of the tries he suggested. You're more than welcome to ask Pierfrancesco about the chat log, don't mind at all. If you ever needed anything about my code as I said before feel free to contact me at matteo.bertocchi6@studio.unibo.it. I wish you the best of luck and hope youo'll achieve great results even for the more complex cases! :3

Good luck and have fun on this work, and if possible thank Pasquale and Pierfrancesco once more for everything they've done for me! ^^

Matte

-----------------------

The gitlab folder of the thesis is here: https://gitlab.com/ADeGobbis/pinn-electrophysiology/-/tree/master

Aliev-Panfilov tries: https://docs.google.com/document/d/1aXThCN4emnIQ4x2dluRbDobrh3QyOX4sZP_hohkQY3g/edit?usp=sharing

1D Equation (4.5) tries: https://docs.google.com/document/d/1RHCPNm9SGHFJ-OWBmQA_ltUlGFtgMIAVt2-7C2PyYxI/edit?usp=sharing