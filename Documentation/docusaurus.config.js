// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'SimplerLLM',
  tagline: 'Easy Pass to Advanced AI',
  favicon: 'img/Logo.png',

  // Set the production url of your site here
  url: 'https://docs.simplerllm.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'LearnWithHasan', // Usually your GitHub org/user name.
  projectName: 'SimplerLLM', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/hassancs91/SimplerLLM/tree/main/',
        },
        blog: false,
        //blog: {
          // showReadingTime: true,
          // feedOptions: {
          //   type: ['rss', 'atom'],
          //   xslt: true,
          // },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
        //   editUrl:
        //     'https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/',
        //   // Useful options to enforce blogging best practices
        //   onInlineTags: 'warn',
        //   onInlineAuthors: 'warn',
        //   onUntruncatedBlogPosts: 'warn',
        // },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/Logo.png',
      navbar: {
        title: 'SimplerLLM Documention',
        logo: {
          alt: 'SimplerLLM Logo',
          src: './img/Logo.png',
        },
        items: [
          // {
          //   type: 'docSidebar',
          //   sidebarId: 'tutorialSidebar',
          //   position: 'left',
          //   label: 'Tutorial',
          // },
          //{to: '/blog', label: 'Blog', position: 'left'},
          {
            href: 'https://simplerllm.com/',
            label: 'Home Page',
            position: 'left',
          },
          {
            href: 'https://github.com/hassancs91/SimplerLLM/blob/main/readme.md',
            label: 'GitHub',
            position: 'right',
          },
          {
            href: 'https://discord.com/invite/HUrtZXyp3j',
            label: 'Discord',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        // links: [
        //   {
        //     title: 'Navigation',
        //     items: [
        //       {
        //         label: 'Introduction',
        //         to: '/',
        //       },
        //       {
        //         label: 'Home Page',
        //         to: 'https://simplerllm.com',
        //       },
        //     ],
        //   },
        //   {
        //     title: 'Community',
        //     items: [
        //       {
        //         label: 'GitHub',
        //         href: 'https://github.com/hassancs91/SimplerLLM/blob/main/readme.md',
        //       },
        //       {
        //         label: 'Discord',
        //         href: 'https://discord.com/invite/HUrtZXyp3j',
        //       },
        //     ],
        //   },
          // {
          //   title: 'More',
          //   items: [
          //     {
          //       label: 'Blog',
          //       to: '/blog',
          //     },
          //     {
          //       label: 'GitHub',
          //       href: 'https://github.com/facebook/docusaurus',
          //     },
          //   ],
          // },
        //],
        copyright: `Copyright © ${new Date().getFullYear()} SimplerLLM. All Rights Reserved`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
