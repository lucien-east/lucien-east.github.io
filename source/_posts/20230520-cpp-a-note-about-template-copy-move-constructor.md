---
title: "[C++] A Note about Template Copy/Move Constructor"
date: 2023-05-20 13:55:07
categories: [Programming Language, C++]
tags: [C++, Optimisation]
lang: en
mathjax:
---


In C++, even though a function generated from a function template has the same name and the same type as the ordinary function, they are **never equivalent**. Besides, the [non-template function is more preferred](https://en.cppreference.com/w/cpp/language/overload_resolution#Best_viable_function):

```C++ =
template <typename T>
void foo(T) { std::cout << "template foo()" << std::endl; }
void foo(int) { std::cout << "ordinary foo()" << std::endl; }

int main() {
    foo(5);     // print "oridinary foo()"
    return 0;
}
```

On top of that, it is not easy to templify a copy/move constructor because the compiler may implicitly define a copy/move constructor.

<!--more-->

## Example

```C++ =
class Obj {
public:
    Obj() {}
    template <typename T>
    Obj (const T&) { std::cout << "template copy ctor" << std::endl; }
};
```

It still uses the predefined copy constructor:

```C++ =
Obj a;
Obj b{a};   // use the predefined copy constructor rather than the template one
```

**The reason is that a member function template is never a [special member function](https://en.cppreference.com/w/cpp/language/member_functions) and can be ignored when the latter is needed**. Taking the above for an instance, the implicitly generated copy constructor is chosen.


## Deleting Predefined Copy Constructor

One may want to delete the the predefined copy constructor: 

```C++ =
class Obj {
public:
    Obj() {}

    Obj(const Obj&) = delete;   // copying Obj results in an error

    template <typename T>
    Obj (const T&) { std::cout << "template copy ctor" << std::endl; }
};
```

However, this would result in an error when trying to copy `Obj`:

```C++ =
Obj a;
Obj b{a};   // error: call to deleted constructor of 'Obj'
```

## `const volatile` Trick

There is a tricky solution for this: deleting the copy constructor with `const volatile` type. With this, it prevents another copy constructor from being implicitly generated[^1] and the template copy constuctor can be preferred over the deleted copy constructor for **non-volatile** types:

```C++ =
class Obj {
public:
    Obj() {}

    // the predefined copy constructor is deleted
    // with conversion to volatile to enable better match
    Obj(const volatile Obj&) = delete;

    template <typename T>
    Obj (const T&) { std::cout << "template copy ctor" << std::endl; }
};
```

The overload resolution candidates now are `Obj(const volatile Obj&)` and `Obj<Obj&>(const Obj&)`, and the latter is a better match:

```C++ =
Obj a;
Obj b{a};   // call to the templified version: print "template copy ctor"
```

---
Similarly, we can templify move constructor or other special member functions by deleting the predefined special member functions for `const volatile` type.


```C++ =
class Obj {
public:
    Obj() {}
    
    Obj (const volatile Obj&&) = delete;

    template <typename T>
    Obj (T&&) { std::cout << "template move ctor" << std::endl; }
};

int main() {
    Obj a;
    Obj c{std::move(a)};  // print "template move ctor"
    return 0;
}
```


> Note that this still leads to error if we try to operate `Obj` with volatile type. Fortunately, it is rarely used.


[^1]: A non-template constructor for class X is a copy constructor if its first parameter is of type `X&`, `const X&`, `volatile X&` or `const volatile X&`