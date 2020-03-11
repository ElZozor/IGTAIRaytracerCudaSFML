#include "cuda_headers.h"
#include <SFML/Graphics.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iostream>

#include "Intersections.cuh"
#include "scenes_init.h"


static bool resolutionUpdated = false;


inline void updateScene(sf::RenderWindow& window, Scene* scene, bool pause)
{
	static sf::Texture texture;
	static sf::Sprite sprite;
	static bool initialized = false;

	if (!initialized)
	{
		texture.create(scene->camera.width, scene->camera.height);
		sprite.setTexture(texture);

		initialized = true;
	}

	if (pause) return;

	const sf::Uint8* pixels = renderScene(scene);
	texture.update(pixels);


	window.clear(sf::Color::Blue);
	window.draw(sprite);
	window.display();
}



inline void updateSceneFromInput(sf::RenderWindow& window, Scene* scene, sf::Vector2i& mousePosition, const float elapsedTime, bool pause)
{
	using namespace sf; typedef Keyboard KB;
	static float movementSpeed;
	static bool fullscreen;
	static sf::Clock lastModeChange;

	if (pause) return;

	movementSpeed = (KB::isKeyPressed(KB::LShift)) ? 3.f : 1.f;

	if (KB::isKeyPressed(KB::Z))
		scene->camera.position += scene->camera.zdir * elapsedTime * movementSpeed;

	if (KB::isKeyPressed(KB::S))
		scene->camera.position -= scene->camera.zdir * elapsedTime * movementSpeed;

	if (KB::isKeyPressed(KB::Q))
		scene->camera.position -= scene->camera.xdir * elapsedTime * movementSpeed;

	if (KB::isKeyPressed(KB::D))
		scene->camera.position += scene->camera.xdir * elapsedTime * movementSpeed;

	if (KB::isKeyPressed(KB::Space))
		scene->camera.position += scene->camera.ydir * elapsedTime * movementSpeed;

	if (KB::isKeyPressed(KB::LControl))
		scene->camera.position -= scene->camera.ydir * elapsedTime * movementSpeed;


	sf::Vector2i offset = sf::Mouse::getPosition() - mousePosition;
	if (offset.y != 0)
		rotateCamera(&scene->camera, scene->camera.xdir, glm::radians(float(offset.y) / float(s_height) * 180.f));

	if (offset.x != 0)
		rotateCamera(&scene->camera, glm::vec3(0, 1, 0), glm::radians(float(offset.x) / float(s_width) * 180.f));

	sf::Mouse::setPosition(window.getPosition() + sf::Vector2i(s_width / 2, s_height / 2));


	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Enter) && sf::Keyboard::isKeyPressed(sf::Keyboard::Enter))
	{
		if (lastModeChange.getElapsedTime().asSeconds() > 0.5f)
		{
			fullscreen = !fullscreen;
			const auto style = (fullscreen) ? sf::Style::Fullscreen : sf::Style::Default;
			window.create(sf::VideoMode(window.getSize().x, window.getSize().y), "IGTAI Raytracer CUDA", style);
			window.setMouseCursorVisible(false);

			mousePosition = window.getPosition() + sf::Vector2i(s_width, s_height) / 2;

			lastModeChange.restart();
		}
	}
}



int main(int argc, char** argv)
{
	Scene* scene;
	int sceneNumber;
	if (argc < 2)
	{
		std::cout << "Scene :";
		std::cin >> sceneNumber;
	}
	else
	{
		sceneNumber = std::stoi(argv[1]);
	}

	scene = initScene(sceneNumber);


	sf::RenderWindow window(sf::VideoMode(s_width, s_height), "IGTAI Raytracer SFML");
	window.setMouseCursorVisible(false);

	sf::Vector2i mousePosition = window.getPosition() + sf::Vector2i(s_width, s_height) / 2;
	sf::Mouse::setPosition(mousePosition);

	sf::Clock clock;
	sf::Event event;
	bool hasFocus = true;
	bool pause = false;
	bool fullscreen = false;
	while (window.isOpen())
	{
		if (!window.hasFocus())
		{
			hasFocus = false;
			continue;
		}

		if (!hasFocus)
		{
			sf::Mouse::setPosition(mousePosition);
			hasFocus = true;
		}
	
		float elapsedTime = clock.restart().asSeconds();
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
			{
				window.close();
				return 0;
			}

			if (event.type == sf::Event::KeyPressed)
			{
				switch (event.key.code)
				{
				case sf::Keyboard::Escape:
					pause = !pause;

					if (!pause)
					{
						sf::Mouse::setPosition(mousePosition);
					}

					window.setMouseCursorVisible(pause);

					break;
				}
			}
		}

		updateSceneFromInput(window, scene, mousePosition, elapsedTime, pause);
		updateScene(window, scene, pause);
	}

	return 0;
}