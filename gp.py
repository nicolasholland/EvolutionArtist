from random import random,randint,choice
from copy import deepcopy
from math import log

################## Node Classes
# Node Classes are taken from the book "Collective Intelligence" by Toby Segaran

class fwrapper:
  def __init__(self,function,childcount,name):
    self.function=function
    self.childcount=childcount
    self.name=name

class node:
  def __init__(self,fw,children):
    self.function=fw.function
    self.name=fw.name
    self.children=children
  def evaluate(self,inp):
    results=[n.evaluate(inp) for n in self.children]
    return self.function(results)

  def display(self,indent=0,depth=0):
    print (' '*indent)+self.name, depth
    for c in self.children:
      c.display(indent+1,depth+1)

class paramnode:
  def __init__(self,idx):
    self.idx=idx

  def evaluate(self,inp):
    return inp[self.idx]

  def display(self,indent=0):
    print '%sp%d' % (' '*indent,self.idx)

class constnode:
  def __init__(self,v):
    self.v=v
  def evaluate(self,inp):
    return self.v
  def display(self,indent=0, depth=0):
    print '%s%d' % (' '*indent,self.v)


class colornode:
  def __init__(self,v):
    self.v=v
  def evaluate(self,inp):
    if self.v > 255:
      return 255
    if self.v < 0:
      return 0
    return self.v
  def display(self,indent=0,depth=0):
    print '%s%d' % (' '*indent,self.v)


################## image nodes
import cv2
import numpy as np
from matplotlib import pyplot as plt


HEIGHT = 200
WIDTH  = 200

# create empty image (all white)
def init(l):
  empty = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
  empty[:,:] = (255, 255, 255)
  return empty
initw = fwrapper(init, 1, 'init')

# draw circle into image
def circle(l):
  cv2.circle(l[0],(l[1],l[2]),l[3],(l[4],l[5],l[6]),-1)
  return l[0]
circlew = fwrapper(circle,6,'circle')

# gaussian blur
def blur(l):
  return cv2.GaussianBlur(l[0],(5,5),0)
blurw = fwrapper(blur, 1, 'blur')

# draw line into image
def line(l):
  cv2.line(l[0],(l[1],l[2]),(l[3],l[4]),(l[5],l[6],l[7]),l[8])
  return l[0]
linew = fwrapper(line,9,'line')
  
flist=[circlew,linew,blurw]

def testtree():
  return  node(linew,
            [node(circlew,[node(initw,[]), constnode(50), constnode(50),constnode(5), colornode(255),colornode(0),colornode(0)]),
              constnode(70),constnode(170),constnode(160),constnode(50),colornode(0),
              colornode(0),colornode(255),constnode(5)])

# show image
def si(img):
  plt.imshow(img)
  plt.axis('off')
  plt.show()

# show multiple images
def sim(img1, img2, img3, img4, img5, img6, img7, img8, img9):
  plt.subplot(3,3,1),plt.imshow(img1),plt.axis('off')
  plt.subplot(3,3,2),plt.imshow(img2),plt.axis('off')
  plt.subplot(3,3,3),plt.imshow(img3),plt.axis('off')
  plt.subplot(3,3,4),plt.imshow(img4),plt.axis('off')
  plt.subplot(3,3,5),plt.imshow(img5),plt.axis('off')
  plt.subplot(3,3,6),plt.imshow(img6),plt.axis('off')
  plt.subplot(3,3,7),plt.imshow(img7),plt.axis('off')
  plt.subplot(3,3,8),plt.imshow(img8),plt.axis('off')
  plt.subplot(3,3,9),plt.imshow(img9),plt.axis('off')
  plt.show()

##################### Tree
from ArtCritic import ArtCritic, critic2

def makerandomtree(img, maxdepth=50,fpr=0.99,ppr=0.6):
  if random( )<fpr and maxdepth>0:
    f=choice(flist)
    children=[makerandomtree(img,maxdepth-1)]    
    children=randomparams(children, f)
    return node(f,children)
  else:
    return img

def randomparams(c, f):
  # circle params
  if f.name == 'circle':
    c.append(constnode(randint(0,HEIGHT)))
    c.append(constnode(randint(0,HEIGHT)))
    c.append(constnode(randint(0,10)))
    for i in range(0,3):
      c.append(colornode(randint(0,255)))
  # line params
  if f.name == 'line':
    for i in range(0,4):
      c.append(constnode(randint(0,HEIGHT)))
    for i in range(0,3):
      c.append(colornode(randint(0,255)))
    c.append(constnode(randint(0,10)))
  return c

def maketree():  
  img = node(initw,[])
  return makerandomtree(img)

def scorefunction(tree):
 #return ArtCritic(tree.evaluate(1), 'resizedArt')
 return critic2(tree.evaluate(1), 'resizedArt')

##################### Mutation

def mutate(t,img,probchange=0.1):
  if random( )<probchange:
    return maketree()
  else:
    result=deepcopy(t)
    try:
      result.children[0]=mutate(result.children[0] ,img,probchange)
    except:
      pass
    return result

def crossover(t1,t2,probswap=0.7,top=1):
  if random( )<probswap and not top: 
    return deepcopy(t2)
  else:
    result=deepcopy(t1)
    try:    
      result.children[0] = crossover(t1.children[0],t2.children[0],probswap,0)
    except:
      pass
    return result


###################### Evolution

def getrankfunction():
  def rankfunction(population):
    scores=[(scorefunction(t),t) for t in population]
    scores.sort( )
    return scores
  return rankfunction

def evolve(pc,popsize,rankfunction,maxgen=500,
    mutationrate=0.1,breedingrate=0.4,pexp=0.3,pnew=0.05):
  # Returns a random number, tending towards lower numbers. The lower pexp
  # is, more lower numbers you will get
  def selectindex( ):
    return int(log(random( ))/log(pexp))

  # Create a random initial population
  population=[maketree() for i in range(popsize)]
  for i in range(maxgen):
    scores=rankfunction(population)

    print scores[0][0]," iteration:",i
    # Show best 9 images of current generation
    #sim(scores[0][1].evaluate(1),scores[1][1].evaluate(1),scores[2][1].evaluate(1),
    #    scores[3][1].evaluate(1),scores[4][1].evaluate(1),scores[5][1].evaluate(1),
    #    scores[6][1].evaluate(1),scores[7][1].evaluate(1),scores[8][1].evaluate(1))

    if scores[0][0]==0: break

    # The two best always make it
    newpop=[scores[0][1],scores[1][1]]
    # Build the next generation
    while len(newpop)<popsize:
      if random()>pnew:
        newtree = mutate(
              crossover(scores[selectindex()][1],
              scores[selectindex()][1],
              probswap=breedingrate),
              pc,probchange=mutationrate)
        newpop.append(newtree)
      else:
        # Add a random node to mix things up
        newtree = maketree()
        newpop.append(newtree)

    population=newpop
  scores[0][1].display( )
  return scores[0][1]
  



#a = gp.evolve(1, 500, rf, maxgen=500, mutationrate=0.9, breedingrate= 0.9, pexp=0.3, pnew=0.8)





