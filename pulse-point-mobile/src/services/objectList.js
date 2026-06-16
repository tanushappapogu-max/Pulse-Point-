/**
 * objectList.js — Comprehensive hardcoded object list for offline fallback
 *
 * Used when the CNN server is unreachable. Provides autocomplete suggestions
 * and validates that a search target is a real-world object.
 */

export const KNOWN_OBJECTS = [
  // ── Furniture ─────────────────────────────────────────────────────────────
  'chair', 'armchair', 'recliner', 'sofa', 'couch', 'loveseat', 'sectional',
  'ottoman', 'bench', 'stool', 'barstool', 'desk', 'table', 'dining table',
  'coffee table', 'side table', 'end table', 'nightstand', 'dresser',
  'chest of drawers', 'wardrobe', 'armoire', 'bookshelf', 'bookcase',
  'shelving unit', 'cabinet', 'filing cabinet', 'credenza', 'sideboard',
  'buffet', 'tv stand', 'entertainment center', 'bed', 'twin bed', 'full bed',
  'queen bed', 'king bed', 'bunk bed', 'daybed', 'futon', 'crib', 'cot',
  'mattress', 'headboard', 'footboard', 'bed frame', 'ladder', 'step stool',

  // ── Bedroom items ─────────────────────────────────────────────────────────
  'pillow', 'blanket', 'comforter', 'duvet', 'quilt', 'bedsheet', 'pillowcase',
  'alarm clock', 'lamp', 'floor lamp', 'desk lamp', 'bedside lamp', 'mirror',
  'full-length mirror', 'vanity', 'vanity mirror', 'jewelry box', 'jewelry',
  'necklace', 'bracelet', 'ring', 'earrings', 'watch', 'hair brush',
  'hair dryer', 'curling iron', 'flat iron', 'comb', 'razor', 'shaver',

  // ── Living room ───────────────────────────────────────────────────────────
  'tv', 'television', 'monitor', 'projector', 'screen', 'remote control',
  'remote', 'speaker', 'bluetooth speaker', 'soundbar', 'subwoofer',
  'stereo system', 'record player', 'vinyl record', 'dvd player',
  'blu-ray player', 'gaming console', 'xbox', 'playstation', 'nintendo switch',
  'game controller', 'joystick', 'vr headset', 'picture frame', 'painting',
  'artwork', 'sculpture', 'vase', 'candle', 'candle holder', 'clock',
  'wall clock', 'grandfather clock', 'mantel clock', 'fireplace', 'fan',
  'ceiling fan', 'air purifier', 'humidifier', 'dehumidifier', 'heater',
  'space heater', 'air conditioner', 'window ac unit', 'plant', 'houseplant',
  'flower pot', 'planter', 'rug', 'carpet', 'curtain', 'blinds', 'drape',
  'window', 'door', 'doormat', 'throw pillow', 'throw blanket',

  // ── Kitchen items ─────────────────────────────────────────────────────────
  'refrigerator', 'fridge', 'freezer', 'stove', 'oven', 'microwave',
  'toaster', 'toaster oven', 'dishwasher', 'sink', 'faucet', 'kettle',
  'electric kettle', 'coffee maker', 'coffee machine', 'espresso machine',
  'french press', 'blender', 'food processor', 'mixer', 'stand mixer',
  'hand mixer', 'instant pot', 'slow cooker', 'crockpot', 'air fryer',
  'rice cooker', 'waffle maker', 'sandwich maker', 'juicer', 'can opener',
  'bottle opener', 'corkscrew', 'grater', 'peeler', 'strainer', 'colander',
  'cutting board', 'knife', 'chef knife', 'bread knife', 'paring knife',
  'kitchen scissors', 'spatula', 'ladle', 'tongs', 'whisk', 'rolling pin',
  'measuring cup', 'measuring spoon', 'mixing bowl', 'bowl', 'plate',
  'dinner plate', 'salad plate', 'cup', 'mug', 'glass', 'wine glass',
  'champagne glass', 'shot glass', 'pitcher', 'jug', 'thermos', 'water bottle',
  'pot', 'pan', 'frying pan', 'saucepan', 'stockpot', 'wok', 'baking sheet',
  'baking pan', 'cake pan', 'muffin tin', 'loaf pan', 'casserole dish',
  'roasting pan', 'dutch oven', 'skillet', 'griddle', 'oven mitt', 'apron',
  'dish towel', 'sponge', 'dish rack', 'utensil holder', 'spice rack',
  'paper towel', 'aluminum foil', 'plastic wrap', 'ziplock bag', 'trash can',
  'recycling bin', 'compost bin',

  // ── Food items ────────────────────────────────────────────────────────────
  'apple', 'banana', 'orange', 'lemon', 'lime', 'grape', 'strawberry',
  'blueberry', 'raspberry', 'watermelon', 'cantaloupe', 'pineapple', 'mango',
  'peach', 'pear', 'plum', 'cherry', 'avocado', 'tomato', 'potato',
  'sweet potato', 'carrot', 'broccoli', 'cauliflower', 'onion', 'garlic',
  'pepper', 'bell pepper', 'cucumber', 'zucchini', 'eggplant', 'spinach',
  'lettuce', 'kale', 'celery', 'asparagus', 'corn', 'mushroom', 'bread',
  'loaf of bread', 'baguette', 'bagel', 'muffin', 'croissant', 'tortilla',
  'rice', 'pasta', 'noodles', 'cereal', 'oatmeal', 'egg', 'eggs', 'milk',
  'butter', 'cheese', 'yogurt', 'cream', 'juice', 'coffee', 'tea', 'water',
  'soda', 'beer', 'wine', 'chocolate', 'candy', 'cookie', 'cake', 'pie',
  'ice cream', 'chips', 'crackers', 'peanut butter', 'jam', 'honey',
  'ketchup', 'mustard', 'mayonnaise', 'hot sauce', 'olive oil', 'vinegar',
  'salt', 'pepper shaker', 'sugar', 'flour', 'canned food', 'soup can',

  // ── Bathroom items ────────────────────────────────────────────────────────
  'toothbrush', 'toothpaste', 'mouthwash', 'floss', 'soap', 'hand soap',
  'shampoo', 'conditioner', 'body wash', 'lotion', 'sunscreen', 'deodorant',
  'perfume', 'cologne', 'makeup', 'lipstick', 'mascara', 'foundation',
  'eye shadow', 'blush', 'nail polish', 'nail clippers', 'tweezers',
  'cotton swabs', 'cotton balls', 'toilet paper', 'towel', 'bath towel',
  'hand towel', 'washcloth', 'bath mat', 'shower curtain', 'toilet',
  'toilet brush', 'plunger', 'scale', 'bathroom scale', 'first aid kit',
  'bandage', 'medicine', 'pill bottle', 'thermometer',

  // ── Electronics & tech ────────────────────────────────────────────────────
  'laptop', 'computer', 'desktop computer', 'tablet', 'ipad', 'phone',
  'smartphone', 'iphone', 'charger', 'phone charger', 'laptop charger',
  'power bank', 'usb cable', 'hdmi cable', 'ethernet cable', 'headphones',
  'earbuds', 'airpods', 'keyboard', 'mouse', 'mousepad', 'webcam',
  'microphone', 'printer', 'scanner', 'external hard drive', 'usb drive',
  'flash drive', 'sd card', 'router', 'modem', 'extension cord',
  'power strip', 'surge protector', 'smart home hub', 'alexa', 'google home',
  'smart bulb', 'smart plug', 'doorbell camera', 'security camera',
  'baby monitor', 'calculator', 'e-reader', 'kindle', 'smartwatch',
  'fitness tracker', 'gps device', 'drone', 'camera', 'dslr camera',
  'mirrorless camera', 'film camera', 'polaroid camera', 'tripod', 'lens',

  // ── Office & school supplies ──────────────────────────────────────────────
  'pen', 'pencil', 'marker', 'highlighter', 'eraser', 'ruler', 'stapler',
  'staples', 'tape', 'scissors', 'paper clips', 'binder clips', 'rubber bands',
  'pushpins', 'sticky notes', 'notebook', 'binder', 'folder', 'envelope',
  'stamps', 'paper', 'printer paper', 'index cards', 'whiteboard', 'chalkboard',
  'calendar', 'planner', 'desk organizer', 'inbox tray', 'tape dispenser',
  'hole puncher', 'shredder', 'desk', 'office chair', 'whiteboard marker',
  'book', 'textbook', 'dictionary', 'magazine', 'newspaper', 'journal',

  // ── Clothing & accessories ────────────────────────────────────────────────
  'shirt', 't-shirt', 'polo shirt', 'dress shirt', 'blouse', 'sweater',
  'hoodie', 'sweatshirt', 'jacket', 'coat', 'blazer', 'suit jacket', 'vest',
  'pants', 'jeans', 'shorts', 'leggings', 'skirt', 'dress', 'suit', 'tuxedo',
  'tie', 'bow tie', 'belt', 'suspenders', 'underwear', 'bra', 'socks',
  'stockings', 'tights', 'pajamas', 'robe', 'swimsuit', 'bikini', 'swim trunks',
  'shoes', 'sneakers', 'running shoes', 'boots', 'heels', 'flats', 'sandals',
  'flip flops', 'loafers', 'dress shoes', 'slippers', 'hat', 'baseball cap',
  'beanie', 'sun hat', 'fedora', 'gloves', 'mittens', 'scarf', 'sunglasses',
  'glasses', 'reading glasses', 'contact lenses', 'bag', 'purse', 'handbag',
  'tote bag', 'backpack', 'briefcase', 'wallet', 'clutch', 'duffel bag',
  'suitcase', 'luggage', 'umbrella', 'rain jacket', 'raincoat', 'rainboots',

  // ── Sports & fitness ─────────────────────────────────────────────────────
  'basketball', 'football', 'soccer ball', 'baseball', 'softball', 'tennis ball',
  'golf ball', 'volleyball', 'bowling ball', 'rugby ball', 'frisbee', 'hockey puck',
  'basketball hoop', 'goal post', 'net', 'tennis racket', 'badminton racket',
  'ping pong paddle', 'golf club', 'baseball bat', 'cricket bat', 'hockey stick',
  'lacrosse stick', 'bowling pin', 'skateboard', 'longboard', 'roller skates',
  'ice skates', 'ski', 'snowboard', 'surfboard', 'paddleboard', 'kayak',
  'canoe', 'bicycle', 'bike', 'mountain bike', 'road bike', 'treadmill',
  'stationary bike', 'elliptical', 'rowing machine', 'weight bench',
  'barbell', 'dumbbell', 'kettlebell', 'resistance band', 'jump rope',
  'yoga mat', 'foam roller', 'pull-up bar', 'punching bag', 'boxing gloves',
  'helmet', 'knee pads', 'shin guards', 'cleats', 'athletic bag',
  'water bottle', 'protein shaker', 'gym bag',

  // ── Tools & hardware ─────────────────────────────────────────────────────
  'hammer', 'screwdriver', 'wrench', 'pliers', 'drill', 'power drill',
  'saw', 'circular saw', 'jigsaw', 'handsaw', 'level', 'tape measure',
  'measuring tape', 'utility knife', 'box cutter', 'chisel', 'mallet',
  'nail', 'screw', 'bolt', 'nut', 'washer', 'sandpaper', 'paint brush',
  'paint roller', 'paint tray', 'spray paint', 'paint can', 'caulk gun',
  'staple gun', 'heat gun', 'soldering iron', 'multimeter', 'flashlight',
  'work light', 'extension ladder', 'step ladder', 'toolbox', 'tool belt',
  'safety goggles', 'work gloves', 'hard hat', 'ear protection', 'dust mask',
  'vacuum', 'shop vac', 'leaf blower', 'power washer',

  // ── Cleaning supplies ─────────────────────────────────────────────────────
  'broom', 'dustpan', 'mop', 'bucket', 'vacuum cleaner', 'robot vacuum',
  'steam mop', 'window squeegee', 'cleaning spray', 'bleach', 'detergent',
  'laundry detergent', 'fabric softener', 'dryer sheet', 'washing machine',
  'dryer', 'iron', 'ironing board', 'lint roller', 'clothes rack',
  'clothes hanger', 'laundry basket', 'hamper', 'drying rack',

  // ── Garden & outdoor ─────────────────────────────────────────────────────
  'lawn mower', 'hedge trimmer', 'weed whacker', 'leaf blower', 'rake',
  'shovel', 'trowel', 'hoe', 'pitchfork', 'garden hose', 'sprinkler',
  'watering can', 'wheelbarrow', 'flower pot', 'garden gloves', 'pruning shears',
  'seed packet', 'fertilizer', 'mulch', 'compost bin', 'bird feeder',
  'bird bath', 'garden gnome', 'outdoor furniture', 'patio chair',
  'adirondack chair', 'hammock', 'outdoor table', 'picnic table', 'grill',
  'bbq grill', 'charcoal', 'propane tank', 'fire pit', 'patio umbrella',
  'outdoor rug', 'garden statue', 'wind chime', 'solar light', 'outdoor light',
  'mailbox', 'fence', 'gate', 'shed', 'garage', 'garage door',
  'trash can', 'recycling bin', 'compost bin', 'hose reel',

  // ── Vehicles & transport ──────────────────────────────────────────────────
  'car', 'sedan', 'suv', 'truck', 'pickup truck', 'van', 'minivan',
  'station wagon', 'convertible', 'sports car', 'electric car', 'hybrid car',
  'motorcycle', 'scooter', 'moped', 'bicycle', 'electric bike', 'e-scooter',
  'bus', 'school bus', 'subway', 'train', 'trolley', 'tram', 'taxi', 'uber',
  'airplane', 'helicopter', 'boat', 'sailboat', 'yacht', 'jet ski',
  'car key', 'car seat', 'stroller', 'baby carriage', 'shopping cart',
  'dolly', 'hand truck', 'skateboard', 'longboard', 'scooter',

  // ── Animals ───────────────────────────────────────────────────────────────
  'dog', 'cat', 'fish', 'bird', 'hamster', 'rabbit', 'guinea pig', 'turtle',
  'snake', 'lizard', 'parrot', 'canary', 'goldfish', 'horse', 'cow', 'pig',
  'sheep', 'goat', 'chicken', 'duck', 'turkey', 'goose', 'squirrel',
  'chipmunk', 'raccoon', 'deer', 'fox', 'bear', 'wolf', 'lion', 'tiger',
  'elephant', 'giraffe', 'zebra', 'monkey', 'gorilla', 'penguin', 'dolphin',
  'whale', 'shark', 'eagle', 'owl', 'hawk', 'crow', 'pigeon', 'sparrow',
  'ant', 'bee', 'butterfly', 'spider',

  // ── Musical instruments ───────────────────────────────────────────────────
  'guitar', 'acoustic guitar', 'electric guitar', 'bass guitar', 'ukulele',
  'piano', 'keyboard', 'synthesizer', 'drum kit', 'drum', 'snare drum',
  'bass drum', 'cymbal', 'violin', 'cello', 'viola', 'bass', 'trumpet',
  'trombone', 'tuba', 'saxophone', 'clarinet', 'flute', 'oboe', 'harp',
  'banjo', 'mandolin', 'harmonica', 'accordion', 'xylophone', 'marimba',
  'bongo', 'djembe', 'tambourine', 'cowbell', 'triangle', 'microphone stand',
  'music stand', 'sheet music', 'guitar pick', 'guitar strap', 'guitar case',

  // ── Art & craft ───────────────────────────────────────────────────────────
  'canvas', 'paint', 'acrylic paint', 'watercolor', 'oil paint',
  'colored pencils', 'chalk pastels', 'paintbrush', 'palette', 'easel',
  'sketchbook', 'drawing tablet', 'clay', 'sculpting tools', 'yarn', 'knitting needles',
  'crochet hook', 'sewing machine', 'needle', 'thread', 'fabric', 'scissors',
  'hot glue gun', 'craft paper', 'stencil', 'stamp', 'ink pad', 'washi tape',

  // ── Medical & health ─────────────────────────────────────────────────────
  'wheelchair', 'walker', 'cane', 'crutches', 'hearing aid', 'glasses',
  'contact lens case', 'eye drops', 'blood pressure monitor', 'pulse oximeter',
  'glucometer', 'insulin', 'inhaler', 'nebulizer', 'cpap machine', 'heating pad',
  'ice pack', 'ace bandage', 'splint', 'syringe', 'iv bag', 'stethoscope',
  'hospital bed', 'exam table', 'scale', 'thermometer', 'otoscope',
  'blood pressure cuff', 'surgical mask', 'n95 mask', 'latex gloves',

  // ── Toys & games ─────────────────────────────────────────────────────────
  'lego', 'action figure', 'doll', 'stuffed animal', 'teddy bear', 'puzzle',
  'board game', 'chess set', 'checkers', 'monopoly', 'scrabble', 'jenga',
  'playing cards', 'dice', 'dominos', 'rubiks cube', 'yo-yo', 'frisbee',
  'kite', 'nerf gun', 'water gun', 'toy car', 'remote control car', 'lego set',
  'train set', 'dollhouse', 'puppet', 'magic kit', 'science kit',
  'telescope', 'microscope', 'kaleidoscope', 'slime', 'fidget spinner',
  'fidget cube', 'pop-it', 'marble', 'hula hoop', 'jump rope',

  // ── Bags & containers ─────────────────────────────────────────────────────
  'box', 'cardboard box', 'storage bin', 'plastic container', 'tupperware',
  'mason jar', 'jar', 'bottle', 'tin can', 'bucket', 'basket', 'crate',
  'safe', 'lockbox', 'briefcase', 'attaché case', 'portfolio', 'file box',
  'cooler', 'ice chest', 'thermos', 'lunch box', 'lunchbag',

  // ── Stationery & paper ────────────────────────────────────────────────────
  'letter', 'postcard', 'greeting card', 'invitation', 'ticket', 'receipt',
  'invoice', 'contract', 'document', 'passport', 'id card', 'credit card',
  'gift card', 'business card', 'photo', 'map', 'poster', 'flyer', 'brochure',

  // ── Lighting ─────────────────────────────────────────────────────────────
  'light bulb', 'led strip', 'chandelier', 'pendant light', 'sconce',
  'nightlight', 'grow light', 'uv light', 'lava lamp', 'neon sign',
  'string lights', 'christmas lights', 'lantern', 'torch', 'candelabra',

  // ── Miscellaneous everyday items ──────────────────────────────────────────
  'keys', 'house key', 'car key', 'keychain', 'coin', 'wallet', 'purse',
  'glasses case', 'phone case', 'charger cable', 'headphone case', 'watch case',
  'ring', 'necklace', 'bracelet', 'earrings', 'pin', 'badge', 'lanyard',
  'lighter', 'matches', 'candle', 'incense', 'essential oil', 'diffuser',
  'alarm', 'smoke detector', 'carbon monoxide detector', 'fire extinguisher',
  'first aid kit', 'emergency kit', 'flashlight', 'battery', 'extension cord',
];

/**
 * Filter the object list to those matching a prefix (case-insensitive).
 * Returns up to `limit` results.
 */
export function suggestObjects(prefix, limit = 8) {
  if (!prefix || prefix.length < 2) return [];
  const q = prefix.toLowerCase().trim();
  return KNOWN_OBJECTS.filter(obj => obj.includes(q)).slice(0, limit);
}

/**
 * Returns true if the given name is in the known objects list.
 */
export function isKnownObject(name) {
  const q = (name || '').toLowerCase().trim();
  return KNOWN_OBJECTS.some(obj => obj === q || obj.includes(q));
}
