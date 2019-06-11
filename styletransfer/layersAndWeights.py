# Jeans(200)
params1 = {
'content_image' : 'styles/model1.jpg',
'style_image' : 'styles/jeans.jpg',
'image_size' : 400,
'style_size' : 400,
'content_layer' : 3,
'content_weight' : 8e-2, 
'style_layers' : (1, 4, 6, 7),
'style_weights' : (200000, 800, 15, 3), 
'tv_weight' : 5e-2
}

# Leather1(50)/Leather2(90)
params2 = {
'content_image' : 'styles/model1.jpg',
'style_image' : 'styles/jeans.jpg',
'image_size' : 400,
'style_size' : 400,
'content_layer' : 3,
'content_weight' : 5e-2, 
'style_layers' : (1, 4, 6, 7),
'style_weights' : (300000, 1000, 15, 3), 
'tv_weight' : 5e-2
}

# Artworks
# means(190)/composition(180)
params3 = {
'content_image' : 'styles/model1.jpg',
'style_image' : 'styles/muse.jpg',
'image_size' : 400,
'style_size' : 400,
'content_layer' : 3,
'content_weight' : 3e-2, 
'style_layers' : (1, 4, 6, 7),
'style_weights' : (300000, 1000, 150, 30), 
'tv_weight' : 5e-2
}
